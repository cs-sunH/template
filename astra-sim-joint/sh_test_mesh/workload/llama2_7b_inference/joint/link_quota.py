"""link_quota.py -- WP3a/WP3b：链路 ∪ HBM 端口的动作级准入配额 +
AIMD 链路 Q 控制律（纯函数 + 状态）。

设计依据：《joint机制改造方案_局部统一内存域》§4.2/§4.4（配额与
反馈控制语义权威）、《三机制联合策略_template仓库设计方案》§7
（开关族）；执行计划 C9 卡（流类别三分冻结）与 C10 卡（AIMD 控制律
+ 稳定性套件 + δ_adm 终冻）、冻结项 D3/D4/D5/F2/F7。

配额是**待验证的动作级准入机制**（不是实例掩码）：按硬件、已知流量及
资源模型派生，不固定为每链路两条流。任一判据不满足只把**动作**标记为
不适用（``inapplicable_reason`` 注明资源种类与余量），实例候选集不动。

配额资源集合 = 有向链路 ∪ 实例 HBM 端口：

* 链路粒度 = 逐有向链路，``Q_init(link) = max(1, floor(rho_eff))`` 为
  逐配置硬件派生（F2；本配置 4050/1640 ≈ 2.47 → 2 只是派生结果，不是
  固定常数）。``max(1, .)`` 仅防结构性死锁，不是性能承诺。rho < 1
  配置的域空化判据 = 定价选中率趋 0（argmin 涌现），**不是**配额关闭——
  配额在 rho<1 下仍按 Q_init=1 常开，物理不可行由端口平价门表达。
* 端口粒度 = 逐实例 HBM 端口，实时流准入判据
  ``B_HBM / (u_port + 1) >= r_hat_KV``（r_hat_KV = 负载视图派生的每流
  消费速率，因果、不假设恒满带宽；``u_port`` = 外部活跃 decode 消费流
  数 ``u_port_base`` + 本注册表在册流数）。端口信用**无** ``max(1, .)``
  保底——忙端口即 0（平价门可全部拒绝）。
* "链路区域"不先验定义：域(t) = 经由仍有剩余信用的链路与端口可达的
  实例集合——域是**出清结果非输入**（:func:`reachable_domain`）。

流类别三分（冻结）：

* ``realtime``（实时流 = decode remote-read 读流）：链路门 + 端口平价门；
* ``elastic``（弹性流 = remote-read prefill 首遍读流）：链路门 + 端口
  并发上限（在册流数 + 新增 <= Q_init；端口入 u_port，**不套平价门**）；
* ``oneshot``（一次性传输 = copy/merge/逐出写回/池恢复）：全额计价 +
  链路计数计入占用与预留、端口在册计入 u_port（**不设平价门**，收紧
  后续实时流）——copy、恢复、驱逐及合并属通用治理范围同等纳入（D4）。

  端口并发上限的计数口径（本模块冻结，供 C11 核对）：按本注册表在册
  流（三类合计）计；外部活跃 decode 消费流（u_port_base）属负载视图、
  只进平价门分母、不进并发计数。计数上限 = Q_init（端口侧与链路侧同
  一硬件派生值；N_bulk 同见下）。

merge 义务预留（借还配对）：

* 链路侧：双候选方向路径各 1 槽，owner 字符串 = ``rid#merge-reserve``
  （与 SH owner 约定同构）；service_done（计算/响应完成）时刻方向裁决
  即释放败者侧（:meth:`LinkQuotaTracker.adjudicate_merge_direction`），
  胜者侧到 ``_on_merge_done``（SH:2099-2112）释放
  （:meth:`LinkQuotaTracker.release_merge`）。
* 端口侧：双候选胜者侧各 1 个 bulk-write 名额，封顶 ``N_bulk(port) =
  Q_init``，超限 quota_deferred（wait_reason=quota_port）。N_bulk 冻结
  为 Q_init，不随链路 Q 调整；static/aimd 之分只作用于链路 Q 调整律
  （AIMD 控制律归 C10），端口平价门与 N_bulk 在 static/aimd 下均常开。
* 量级分层（调用方策略，C11/C14 落地；模块只提供双向各 1 槽原语，
  功能上已覆盖翻转分支）：copy/recompute 轮零翻转零预留（轮内逐块
  交接/重算即权威，C13）；remote-read 轮预留量级 ≈ min(两侧保留量)——
  常规分支（基础驻留 > 增量）= 执行侧保留量（无留存型暂存口径下即
  增量本身），翻转分支（增量 > 基础驻留）败者侧为 home、搬运字节 =
  home 侧保留量；部分驻留/多源状态按唯一有效块集合如实披露。预测
  预留不替代容量与数据就绪约束。

quota_deferred 语义（冻结；对照 joint_scheduler.py:156-159 的零候选
fail-closed——``JointSchedulerError`` 会终局 run）：

* 任一判据不满足 → 动作标记不适用（实例不掩码）；**全部动作不可行时
  严禁走零候选 fail-closed**——正确语义 = 请求回 ``pending_admissions``
  （:func:`quota_deferred_requeue`，模块级实现，永不 raise）；仅
  "释放后仍无任何动作可行"才终端 fail-closed 披露（SH 死锁守卫口径）。
* 等待日志 ``joint_admission_wait`` 携带 ``wait_reason ∈ {capacity,
  quota_link, quota_port}`` 分列；等待计入从请求到达到完成的延迟。

AIMD 控制律（C10，D5 冻结；仅 ``mode=aimd``，static 下 Q 恒 = Q_init）：

* **收缩**（MD）：遥测有效速率 < r_KV 等值线 →
  ``Q(link) ← max(1, floor(Q/2))``。减半因子为本卡裁定披露（D5 只
  显式冻结收缩信号 / k / 阈值带）：canonical AIMD 的 MD 支，与本
  模块 ``set_link_quota`` 的下限 1（F2 防结构性死锁）组合；冻结为
  常数（:data:`AIMD_SHRINK_DIVISOR` = 2），不据任何性能结果调整。
  收缩经 :meth:`LinkQuotaTracker.set_link_quota` 落地：仅 aimd 合法、
  在册流 grandfathered（余量负期间封新准入）、代数 bump 全继承。
  Q 已在下限 1 时的收缩信号只记事件（streak 清零）不调整、不 bump。
* **扩张**（AI）：连续 ``T_expand = k × EWMA(流寿命)``（**k = 10**，
  :data:`AIMD_EXPAND_K`）处于**舒适区**后 ``Q += 1``、streak 清零
  重计。舒适区 = 遥测有效速率 >= ``1.2 × r_KV``（阈值带上沿）。
* **扩张冻结窗口**（O6②，:meth:`LinkQuotaTracker.observe_telemetry`
  的 ``allow_expansion=False``）：冻结 additive-increase 扩张与
  expand_capped，并清除此前 quiet_ns；冻结区间时长不计入解冻后的新
  streak。收缩/保持/判定/EWMA 照常。O6 口径：r̂ 代表值
  回退窗口（active_decode 瞬空、SH 侧 r̂ 走代表值 ctx=1，comfort
  阈值 1.2·r̂_空 极低且 ceiling=floor(B/r̂) 按瞬时 r̂ 锚定暴涨）
  由调用方传 False——空窗不扩张、decode 回归不瀑布（解冻后须从零
  连续累计满 T_expand 才扩张）。
* **胀缩阈值带 [r_KV, 1.2×r_KV] 不重合**：收缩触发集
  ``{rate < r_KV}`` 与扩张触发集 ``{rate >= 1.2·r_KV}`` **不相交**，
  带内为死区——既不收缩、也不累计扩张 streak（死区停留打断"连续
  舒适"）。这是"不重合"的可操作读法：两支控制律无公共触发态，
  阈值 chatter 被结构性排除（稳定性断言 (b) 的依据）。边界语义：
  ``rate == r_KV``（等值线上）不收缩（严格小于）；``rate ==
  1.2·r_KV`` 计舒适（含等号）。
* **扩张上界** = ``max(1, floor(B_link / r_KV))``（收缩等值线本身，
  :func:`aimd_expand_ceiling`）：越过该线扩张 ⇒ 满占用时实测速率
  必 < r_KV ⇒ 下一次采样必收缩——自毁性抖动，结构性排除（防发散
  披露）。r_KV > B_link 的极端配置上界仍保 1（与 Q_init 下限同源）。
* **遥测冻结契约**（§2.1；C8 解析、C11 接线）：SH 侧以
  ``{link_id: 实测有效速率}`` 字典传入——value = 该链路在册流的
  实测**每流**有效速率（bytes/ns，与 JCM"链路峰值带宽 / 除数"同
  一量纲的实测对应量），直接与 r_KV 比较；无在册流的链路不出现在
  字典中（无测量即无信号，§4.1 因果口径；普通缺测遵循 K7/P2-7：
  quiet_ns 保留、缺测 dt 不计、重观测首个样本的 dt 不计入；显式
  ``allow_expansion=False`` 冻结会清除 quiet_ns）。
  模块接口 :meth:`LinkQuotaTracker.observe_telemetry` 与该契约同构，
  单测以合成遥测字典驱动。
* **流寿命 EWMA**（平滑规则冻结）：流 settle（准入与释放均带
  ``now_ns`` 时标）按 ``EWMA ← (1−α)·EWMA + α·lifetime`` 更新，
  α = 1/8（:data:`FLOW_LIFETIME_EWMA_ALPHA`；TCP SRTT（RFC 6298）
  同域惯例，有效窗口 ≈ 8 样本；冻结常数，不按 workload 调参）；
  首样本直接初始化。零寿命（同时刻准入/释放）= 非正服务时长，
  跳过不计。**冷启动** = 尚无寿命样本时 T_expand 不可定义 → 扩张
  抑制（舒适区不累计 streak），首个样本后恢复——不引入任何配置
  可推导之外的估计器。
* **时间戳合同**：admit/release/telemetry 的 ``now_ns`` 均为调用方
  事件钟（SH dispatch 钟）的显式参数（缺省 None = 不参与 AIMD/EWMA
  计时，簿记与代数语义不变）；各通道局部单调性 fail-closed
  （release 早于 admit、遥测钟倒退均拒），通道间不交叉校验。
* static/aimd 之分**只作用于链路 Q 调整律**；端口平价门与 N_bulk
  在 static/aimd 下均常开（C9 冻结，本卡不改——aimd 收缩链路 Q
  不影响端口判据与 N_bulk）。

SH 衔接面（C11 对接合同，本卡只冻结接口语义、不改 SH）：

* **重试键接口**：SH:1775-1802 的 ``_admit_attempt_epoch`` /
  ``_current_retry_key`` epoch 键重试门扩展一个配额代数分量——
  :meth:`LinkQuotaTracker.quota_retry_key` 返回 ``(quota_epoch,)``，
  流 settle / 预留释放 / 方向裁决 / 配额调整均 bump 该代数；C11 以
  :func:`extend_retry_key` 把它并入既有 ``(kv_ledger_epoch, 候选集
  实例纪元)`` 键，使信用释放后 deferred 请求必获再评估。
* **wait_reason 枚举**：``WAIT_CAPACITY``（SH 既有容量等待）/ 
  ``WAIT_QUOTA_LINK`` / ``WAIT_QUOTA_PORT``（本模块新增两值）。

本模块不读环境变量（开关解析在 ``joint_config.JOINT_QUOTA_MODE``），
不直接改动 KV 账本与图发射；遥测解析归 C8（本模块只消费
``{link_id: 实测有效速率}`` 冻结契约字典）；SH 侧接线归 C11。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Hashable, Mapping, Optional, Sequence

# ============================================================ 开关与枚举 ==

#: 配额模式（F7：缺省 off；static/aimd 以显式臂开启）。
QUOTA_MODES = ("off", "static", "aimd")
QUOTA_OFF = "off"
QUOTA_STATIC = "static"
QUOTA_AIMD = "aimd"

#: 流类别三分（C9 冻结；见模块 docstring）。
FLOW_REALTIME = "realtime"
FLOW_ELASTIC = "elastic"
FLOW_ONESHOT = "oneshot"
FLOW_CLASSES = (FLOW_REALTIME, FLOW_ELASTIC, FLOW_ONESHOT)

#: joint_admission_wait 等待日志的 wait_reason 枚举（C11 接线）。
#: capacity = SH 既有容量等待；quota_link / quota_port = 本模块新增分列。
WAIT_CAPACITY = "capacity"
WAIT_QUOTA_LINK = "quota_link"
WAIT_QUOTA_PORT = "quota_port"
WAIT_REASONS = (WAIT_CAPACITY, WAIT_QUOTA_LINK, WAIT_QUOTA_PORT)

#: merge 预留的链路侧 owner 字符串模板（与 SH owner 约定同构：
#: ``rid#readplan``、``rid#decode#{j}``、``rid#merge`` 之外新增本通道）。
MERGE_RESERVE_OWNER_TEMPLATE = "{rid}#merge-reserve"

#: merge 方向枚举（双候选 = 前向合并腿 / 反向合并腿）。
MERGE_DIRECTION_FORWARD = "forward"
MERGE_DIRECTION_REVERSE = "reverse"
MERGE_DIRECTIONS = (MERGE_DIRECTION_FORWARD, MERGE_DIRECTION_REVERSE)

#: δ_adm 近平局带初值（D5 冻结 = 0）。
#: 物理单位 = ns（与 cost_ns 同域）；来源 = 执行计划 D5：原拟
#: "I / 有效带宽" 与 merge_done 终点内已计价的前向合并腿同数值，构成
#: 重复计费，故初值取 0。终值由 C10 三选一闭合，任一选择须登记物理
#: 单位与来源，不据结果扩大。（C10 终冻决议 = **维持 0**，见
#: /tmp/joint_exec/C10/delta_adm_freeze.md 与执行计划 §4 补遗。）
DELTA_ADM_INITIAL_NS = 0

# ------------------------------------------- AIMD 控制律常数（C10，D5）--

#: 扩张时标系数 k（D5 冻结 = 10）：T_expand = k × EWMA(流寿命)。
#: 物理意义：配额调整（慢回路）比流寿命（快回路/被控对象）慢一个
#: 数量级——时间尺度分离，EWMA→决策→负载反馈回路的防振荡依据
#: （设计文档 §4.4"冻结时间尺度"项）。
AIMD_EXPAND_K = 10

#: 胀缩阈值带上沿因子（D5 冻结 = 1.2）：扩张舒适阈值 = 1.2 × r_KV。
#: 带 [r_KV, 1.2×r_KV] 为死区，收缩触发集 {rate < r_KV} 与扩张触发
#: 集 {rate >= 1.2·r_KV} 不相交（"不重合"，设计文档 §4.4"迟滞规则"
#: 项——迟滞本身不充当稳定性证据，稳定性由 C10 五剧本套件验证）。
AIMD_BAND_UPPER_FACTOR = 1.2

#: 收缩 MD 因子（C10 裁定披露，D5 未显式冻结项）：canonical AIMD 的
#: 减半支 ``Q ← max(1, floor(Q/2))``。来源 = AIMD 域标准控制律（TCP
#: 同域）；冻结为常数，不据任何性能结果调整；与 set_link_quota 的
#: 下限 1（F2 防结构性死锁）组合。
AIMD_SHRINK_DIVISOR = 2

#: 流寿命 EWMA 平滑因子（冻结）：α = 1/8，TCP SRTT（RFC 6298）同域
#: 惯例（有效窗口 ≈ 8 样本）。冻结常数，不按 workload 调参。
FLOW_LIFETIME_EWMA_ALPHA = 0.125

#: 遥测信号三分（胀缩阈值带判据，:func:`aimd_band_signal`）。
AIMD_SIGNAL_SHRINK = "shrink"     # rate < r_KV（带下沿之下 → 收缩）
AIMD_SIGNAL_HOLD = "hold"         # r_KV <= rate < 1.2·r_KV（死区）
AIMD_SIGNAL_COMFORT = "comfort"   # rate >= 1.2·r_KV（舒适 = 可扩张）
AIMD_SIGNALS = (AIMD_SIGNAL_SHRINK, AIMD_SIGNAL_HOLD, AIMD_SIGNAL_COMFORT)

#: 遥测消费动作枚举（observe_telemetry 披露位）。
AIMD_ACTION_SHRINK = "shrink"                  # MD：Q 减半（经 set_link_quota）
AIMD_ACTION_SHRINK_FLOOR = "shrink_floor"      # Q 已在下限 1：记事件不调整
AIMD_ACTION_HOLD = "hold"                      # 死区：streak 清零
AIMD_ACTION_COMFORT = "comfort"                # 舒适：streak 累计中
AIMD_ACTION_COMFORT_COLD_START = "comfort_cold_start"  # 冷启动：扩张抑制
AIMD_ACTION_COMFORT_FROZEN = "comfort_frozen"  # O6②：扩张冻结窗（allow_
#     expansion=False）——舒适判定照常、清空 streak、不评估扩张触发
AIMD_ACTION_EXPAND = "expand"                  # AI：Q += 1（经 set_link_quota）
AIMD_ACTION_EXPAND_CAPPED = "expand_capped"    # 触上界：streak 清零不调整


class LinkQuotaError(ValueError):
    """fail-closed：非法模式/参数、借还失配（双释放/漏释放/重复预留）。"""


# ================================================================ 纯函数 ==


def derive_rho_eff(
    noc_link_bytes_per_ns: float,
    local_hbm_bytes_per_ns: float,
) -> float:
    """rho_eff = B_D2D / B_HBM（逐配置硬件派生，§1.2 解析锚点）。"""
    for name, value in (
            ("noc_link_bytes_per_ns", noc_link_bytes_per_ns),
            ("local_hbm_bytes_per_ns", local_hbm_bytes_per_ns)):
        if not (value > 0) or not math.isfinite(value):
            raise LinkQuotaError(
                f"{name} must be a positive finite rate, got {value!r}")
    return noc_link_bytes_per_ns / local_hbm_bytes_per_ns


def derive_q_init(rho_eff: float) -> int:
    """Q_init(link) = max(1, floor(rho_eff))（F2）。

    逐配置硬件派生，**不固定为每链路两条流**——4050/1640 ≈ 2.47 → 2
    只是本配置的派生结果。``max(1, .)`` 仅防结构性死锁（保证任一链路
    至少容纳 1 条流，避免全图信用为零的先验死锁），不是性能承诺；
    rho < 1 配置同样派生 Q_init = 1，配额不关闭（域空化由定价选中率
    表达）。
    """
    if not math.isfinite(rho_eff) or rho_eff < 0:
        raise LinkQuotaError(
            f"rho_eff must be a finite non-negative ratio, got {rho_eff!r}")
    return max(1, int(math.floor(rho_eff)))


def port_parity_admits(
    local_hbm_bytes_per_ns: float,
    u_port: int,
    r_hat_kv_bytes_per_ns: float,
) -> bool:
    """端口平价门：``B_HBM / (u_port + 1) >= r_hat_KV``（§4.2）。

    ``r_hat_KV`` = 负载视图派生的每流消费速率（因果，不假设恒满带宽）；
    ``u_port`` = 活跃 decode 消费流 + 在册传输/远读流（F4 口径）。端口
    信用**无** ``max(1, .)`` 保底——忙端口即 0（本判据可全拒）。
    """
    if u_port < 0:
        raise LinkQuotaError(
            f"u_port must be non-negative, got {u_port!r}")
    for name, value in (
            ("local_hbm_bytes_per_ns", local_hbm_bytes_per_ns),
            ("r_hat_kv_bytes_per_ns", r_hat_kv_bytes_per_ns)):
        if not (value > 0) or not math.isfinite(value):
            raise LinkQuotaError(
                f"{name} must be a positive finite rate, got {value!r}")
    return local_hbm_bytes_per_ns / (u_port + 1) >= r_hat_kv_bytes_per_ns


def port_parity_headroom(
    local_hbm_bytes_per_ns: float,
    u_port: int,
    r_hat_kv_bytes_per_ns: float,
) -> int:
    """平价门余量（port_snapshot 披露位）：仍可准入的实时流数 k。

    即最大 k >= 0 使 ``B_HBM / (u_port + k) >= r_hat_KV``；k = 0 表示
    忙端口（无保底，§4.2）。
    """
    if not port_parity_admits(
            local_hbm_bytes_per_ns, u_port, r_hat_kv_bytes_per_ns):
        return 0
    # B/(u+k) >= r  <=>  u + k <= B/r  <=>  k <= floor(B/r) - u
    return max(0, int(local_hbm_bytes_per_ns / r_hat_kv_bytes_per_ns) - u_port)


def challenger_flips(
    challenger_cost_ns: int,
    incumbent_cost_ns: int,
    delta_adm_ns: int = DELTA_ADM_INITIAL_NS,
) -> bool:
    """δ_adm 近平局带（D5）：挑战动作须胜在册动作族 >= δ_adm 才翻转。

    裁决式：``flip ⇔ incumbent_cost − challenger_cost >= delta_adm``。
    位于共享决策路径、内部对照臂同开关（同一 δ_adm 值作用于主臂与
    内部对照）。使用合同（C11）：本谓词是既有全候选 argmin 比较的
    **附加门**——调用方先按 ``(cost_ns, order_key)`` 字典序确定在册
    胜者，挑战者仅当标准比较偏好它**且**本谓词为真才翻转。δ_adm = 0
    （冻结初值）时本谓词恒开（margin >= 0 与 challenger <= incumbent
    同真域），组合后 ≡ 既有 argmin 语义（平局由 order_key 裁决），
    不借初值夹带行为变更。
    """
    if delta_adm_ns < 0:
        raise LinkQuotaError(
            f"delta_adm_ns must be non-negative, got {delta_adm_ns!r}")
    return incumbent_cost_ns - challenger_cost_ns >= delta_adm_ns


def aimd_band_signal(
    effective_rate_bytes_per_ns: float,
    r_hat_kv_bytes_per_ns: float,
) -> str:
    """胀缩阈值带判据（D5：带 [r_KV, 1.2×r_KV] 不重合）。

    ``rate < r_KV`` → :data:`AIMD_SIGNAL_SHRINK`（收缩触发集）；
    ``rate >= 1.2·r_KV`` → :data:`AIMD_SIGNAL_COMFORT`（扩张触发
    集）；两者之间为死区 :data:`AIMD_SIGNAL_HOLD`。收缩集与扩张集
    **不相交**（无公共触发态——"不重合"的可操作读法）。边界语义：
    等值线本身（rate == r_KV）不收缩（严格小于）；上沿（rate ==
    1.2·r_KV）计舒适（含等号）。
    """
    for name, value in (
            ("effective_rate_bytes_per_ns", effective_rate_bytes_per_ns),
            ("r_hat_kv_bytes_per_ns", r_hat_kv_bytes_per_ns)):
        if not (value > 0) or not math.isfinite(value):
            raise LinkQuotaError(
                f"{name} must be a positive finite rate, got {value!r}")
    if effective_rate_bytes_per_ns < r_hat_kv_bytes_per_ns:
        return AIMD_SIGNAL_SHRINK
    upper = AIMD_BAND_UPPER_FACTOR * r_hat_kv_bytes_per_ns
    if effective_rate_bytes_per_ns < upper:
        return AIMD_SIGNAL_HOLD
    return AIMD_SIGNAL_COMFORT


def aimd_shrink_quota(quota: int) -> int:
    """收缩律（MD 支）：``Q ← max(1, floor(Q / AIMD_SHRINK_DIVISOR))``。

    减半为 canonical AIMD（裁定披露，见 :data:`AIMD_SHRINK_DIVISOR`）；
    下限 1 与 Q_init 的防结构性死锁下限同源（F2）。
    """
    if isinstance(quota, bool) or not isinstance(quota, int) or quota < 1:
        raise LinkQuotaError(
            f"quota must be an integer >= 1, got {quota!r}")
    return max(1, quota // AIMD_SHRINK_DIVISOR)


def aimd_expand_ceiling(
    noc_link_bytes_per_ns: float,
    r_hat_kv_bytes_per_ns: float,
) -> int:
    """扩张上界 = ``max(1, floor(B_link / r_KV))``（收缩等值线本身）。

    扩张越过该线 ⇒ 满占用时实测每流速率必 < r_KV ⇒ 下一次采样必
    收缩——自毁性抖动，结构性排除（防发散 + 防带窄于步长时的跨带
    单步振荡）。r_KV > B_link 的极端配置 floor 为 0 时仍保 1。
    """
    for name, value in (
            ("noc_link_bytes_per_ns", noc_link_bytes_per_ns),
            ("r_hat_kv_bytes_per_ns", r_hat_kv_bytes_per_ns)):
        if not (value > 0) or not math.isfinite(value):
            raise LinkQuotaError(
                f"{name} must be a positive finite rate, got {value!r}")
    return max(1, int(math.floor(
        noc_link_bytes_per_ns / r_hat_kv_bytes_per_ns)))


def extend_retry_key(base_key: tuple, quota_retry_key: tuple) -> tuple:
    """重试键扩展（C11 对接合同）。

    SH 既有键 = ``(kv_ledger_epoch, ((instance, epoch), ...))``
    （SH:1817-1827 ``_current_retry_key``）；配额分量追加在末位：
    ``full = base + (quota_epoch,)``。配额代数 bump（流 settle / 预留
    释放 / 方向裁决 / 配额调整）使 full 键变化 → 重试门重开，deferred
    请求获再评估。缺省 base = () 时即纯配额键。
    """
    return tuple(base_key) + tuple(quota_retry_key)


def reachable_domain(
    source: Hashable,
    adjacency: Mapping[Hashable, Sequence[Hashable]],
    link_has_credit: Callable[[tuple], bool],
    port_admits: Optional[Callable[[Hashable], bool]] = None,
) -> frozenset:
    """域(t) = 经由仍有剩余信用的链路与端口可达的实例集合（§4.2）。

    **域是出清结果非输入**："链路区域"不先验定义；``link_has_credit``
    按当前信用余量裁决逐有向边 ``(u, v)``，``port_admits``（可选）按
    端口侧判据裁决进入实例 v（缺省恒 True——纯链路口径）。``source``
    恒在域内。本函数只做分析披露（C16 D_feed 口径），不裁剪任何动作/
    实例候选。
    """
    domain = {source}
    frontier = [source]
    while frontier:
        node = frontier.pop()
        for neighbor in adjacency.get(node, ()):
            edge = (node, neighbor)
            if (neighbor not in domain
                    and link_has_credit(edge)
                    and (port_admits is None or port_admits(neighbor))):
                domain.add(neighbor)
                frontier.append(neighbor)
    return frozenset(domain)


# ================================================================= 记录 ==


@dataclass(frozen=True)
class QuotaVerdict:
    """单次动作级准入裁决（链路 ∪ 端口原子判定）。

    ``admitted=False`` 即 quota_deferred：动作标记不适用（实例不掩码），
    ``inapplicable_reason`` 注明资源种类与余量，``wait_reason`` 分列
    quota_link / quota_port（SH joint_admission_wait 日志枚举）。
    """

    admitted: bool
    flow_class: Optional[str] = None
    resource_kind: Optional[str] = None     # "link" | "port" | None
    resource_id: Any = None
    remaining: Optional[int] = None         # 失败资源的信用余量
    wait_reason: Optional[str] = None       # quota_link | quota_port | None
    inapplicable_reason: str = ""

    def as_dict(self) -> dict:
        return {
            "admitted": self.admitted,
            "flow_class": self.flow_class,
            "resource_kind": self.resource_kind,
            "resource_id": self.resource_id,
            "remaining": self.remaining,
            "wait_reason": self.wait_reason,
            "inapplicable_reason": self.inapplicable_reason,
        }


@dataclass(frozen=True)
class QuotaDeferred:
    """quota_deferred 回队记录（§4.2：等待计入请求到达→完成延迟）。"""

    request_id: str
    requeue: bool
    wait_reason: str
    retry_key: tuple
    inapplicable_reasons: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "request_id": self.request_id,
            "requeue": self.requeue,
            "wait_reason": self.wait_reason,
            "retry_key": list(self.retry_key),
            "inapplicable_reasons": dict(self.inapplicable_reasons),
        }


def quota_deferred_requeue(
    request_id: str,
    verdicts_by_action: Mapping[str, QuotaVerdict],
    quota_retry_key: tuple,
    base_retry_key: tuple = (),
) -> QuotaDeferred:
    """全部动作配额不可行时的**回队**裁决（C9 步骤 6 冻结语义）。

    对照 joint_scheduler.py:156-159 的零候选 fail-closed
    （``JointSchedulerError`` 会终局 run）：配额判据导致的"全部动作
    不可行"不是执行协议缺口，**严禁**走该 fail-closed——正确语义 =
    请求回 ``pending_admissions``，重试键扩展配额代数（流 settle /
    预留释放 bump 键 → 重试门重开）。仅"释放后仍无任何动作可行"才由
    SH 死锁守卫做终端 fail-closed 披露。本函数即该语义的模块级实现：
    对合法输入只返回回队记录、不因判据结果 raise（空 verdicts 映射
    的防御分支同样回队不终局）；**调用方 bug 仍 fail-closed**（非
    QuotaVerdict 值 / admitted verdict 混入——K8 订正：原"永不 raise"
    措辞与调用方 bug 守卫矛盾）。

    ``wait_reason`` 取 verdicts 中按 WAIT_REASONS 枚举序的首个非 None
    值（确定性裁决，供 joint_admission_wait 分列）。
    """
    for action, verdict in verdicts_by_action.items():
        if not isinstance(verdict, QuotaVerdict):
            raise LinkQuotaError(
                f"verdicts_by_action[{action!r}] must be QuotaVerdict, "
                f"got {type(verdict).__name__}")
        if verdict.admitted:
            raise LinkQuotaError(
                f"quota_deferred_requeue got an admitted verdict for "
                f"action {action!r}; requeue is only for all-infeasible "
                "sets (caller bug)")
    wait_reason = WAIT_CAPACITY
    for reason in WAIT_REASONS:
        if any(verdict.wait_reason == reason
               for verdict in verdicts_by_action.values()):
            wait_reason = reason
            break
    return QuotaDeferred(
        request_id=request_id,
        requeue=True,
        wait_reason=wait_reason,
        retry_key=extend_retry_key(base_retry_key, quota_retry_key),
        inapplicable_reasons={
            action: verdict.inapplicable_reason
            for action, verdict in verdicts_by_action.items()
        },
    )


# ================================================================ 状态 ==


class FlowLifetimeEwma:
    """EWMA(流寿命)——AIMD 扩张时标 T_expand 的服务因子。

    平滑规则冻结：``EWMA ← (1−α)·EWMA + α·lifetime``，α =
    :data:`FLOW_LIFETIME_EWMA_ALPHA`（1/8，TCP SRTT 同域惯例）；首
    样本直接初始化（寿命样本因果可得、逐流等权进入窗口，不引入
    偏置修正）。只接受**正**服务时长（零/负寿命 = 非有效样本，由
    调用方跳过或 fail-closed）。
    """

    __slots__ = ("_value_ns", "_samples")

    def __init__(self) -> None:
        self._value_ns: Optional[float] = None
        self._samples = 0

    def observe(self, lifetime_ns: float) -> None:
        """喂一个流寿命样本（正有限时长；非法输入 fail-closed）。"""
        if not (lifetime_ns > 0) or not math.isfinite(lifetime_ns):
            raise LinkQuotaError(
                "lifetime_ns must be a positive finite duration, got "
                f"{lifetime_ns!r}")
        if self._value_ns is None:
            self._value_ns = float(lifetime_ns)
        else:
            self._value_ns = (
                (1.0 - FLOW_LIFETIME_EWMA_ALPHA) * self._value_ns
                + FLOW_LIFETIME_EWMA_ALPHA * lifetime_ns)
        self._samples += 1

    @property
    def value_ns(self) -> Optional[float]:
        """当前 EWMA（bytes 流寿命 ns；无样本 = None = 冷启动）。"""
        return self._value_ns

    @property
    def samples(self) -> int:
        return self._samples


@dataclass
class _AimdLinkState:
    """逐链路 AIMD 遥测状态（上次采样时刻/信号 + 连续舒适 streak）。

    K7（P2-7，2026-09-23 外部审计）：``last_seq`` 记录该链路上次出现
    在哪个 observe_telemetry 调用序——普通缺测间隔（无在册流 ⇒ 字典
    缺席）内 quiet_ns 保留，重观测首样本 dt 不计。O6 显式冻结另清除
    quiet_ns；``last_expansion_allowed`` 用于跳过相邻冻结/解冻样本间隔。
    """

    last_ns: Optional[int] = None
    last_signal: Optional[str] = None
    last_seq: Optional[int] = None
    last_expansion_allowed: Optional[bool] = None
    quiet_ns: int = 0     # 连续（无收缩事件且 rate >= 1.2·r_KV）时长


@dataclass
class _FlowEnrollment:
    """一条已准入流在册记录（借还配对的"借"侧）。"""

    owner: str
    flow_class: str
    links: tuple                  # 有向链路序列（可含重复 = 多槽）
    port_id: Optional[Hashable]
    slots: int


@dataclass
class _MergeReserve:
    """一次 merge 义务预留（双向各 1 槽 + 双候选胜者侧各 1 bulk 名额）。"""

    rid: str
    links_forward: tuple
    links_reverse: tuple
    port_forward: Optional[Hashable]
    port_reverse: Optional[Hashable]
    adjudicated: Optional[str] = None   # 胜者方向（service_done 裁决后）

    @property
    def owner(self) -> str:
        return MERGE_RESERVE_OWNER_TEMPLATE.format(rid=self.rid)


class LinkQuotaTracker:
    """链路 ∪ HBM 端口的配额状态机（准入/释放/预留/重试代数）。

    构造即派生 ``rho_eff`` 与 ``Q_init``（F2；不接受外部覆写 Q_init——
    避免非派生常数混入）。``mode=off`` 时全部**门**旁路（admit 恒过）
    但簿记照记——释放配对语义与 mode 无关（调用方无论何种调用节律，
    admit/release 配对始终成立）。``mode=static`` 链路 Q 恒 = Q_init；
    ``mode=aimd`` 由 :meth:`observe_telemetry` 驱动 AIMD 控制律
    （C10 落地：收缩 MD / 扩张 AI / 胀缩阈值带 / 流寿命 EWMA），调整
    经 :meth:`set_link_quota`（防结构性死锁下限 1）。端口平价门与
    N_bulk（= Q_init，冻结）在 static/aimd 下均常开、不随链路 Q 调整。
    """

    def __init__(
        self,
        *,
        mode: str = QUOTA_STATIC,
        noc_link_bytes_per_ns: float,
        local_hbm_bytes_per_ns: float,
        delta_adm_ns: int = DELTA_ADM_INITIAL_NS,
    ) -> None:
        if mode not in QUOTA_MODES:
            raise LinkQuotaError(
                f"mode must be one of {QUOTA_MODES}, got {mode!r}")
        if delta_adm_ns < 0:
            raise LinkQuotaError(
                f"delta_adm_ns must be non-negative, got {delta_adm_ns!r}")
        self._mode = mode
        self._b_hbm = float(local_hbm_bytes_per_ns)
        self._rho_eff = derive_rho_eff(
            noc_link_bytes_per_ns, local_hbm_bytes_per_ns)
        self._q_init = derive_q_init(self._rho_eff)
        self._n_bulk = self._q_init           # N_bulk(port) = Q_init（冻结）
        self._delta_adm_ns = int(delta_adm_ns)
        # 链路侧：Q（aimd 可调）/ 占用（准入流）/ 预留（merge-reserve）。
        self._link_quota: dict = {}
        self._link_occ: dict = {}             # link -> {owner: count}
        self._link_res: dict = {}             # link -> {owner: count}
        # 端口侧：在册流（owner -> (count, flow_class)）与 bulk 名额。
        self._port_enroll: dict = {}          # port -> {owner: (count, cls)}
        self._port_bulk: dict = {}            # port -> {owner: count}
        self._flows: dict = {}                # owner -> _FlowEnrollment
        self._merges: dict = {}               # rid -> _MergeReserve
        # N1（2026-09-23 复核审计1）：同事务借槽账本——merge 预留在
        # 同 rid 流已占用的链路上不再重复计入 reserved（读流与 merge
        # 写是同一事务的时序先后阶段，同事务同链峰值 = max 而非 sum；
        # 跨事务叠加不变——他事务预留仍全额入 reserved）。读流 settle
        # 时借槽**转移**为 reserved（occ→res，总计数不变、无新准入
        # 窗口）；预留先撤时借记就地消解（_release_merge_side）。
        self._merge_borrow: dict = {}         # rid -> {link: borrowed}
        # O10②（2026-09-23 终轮深挖）：借槽守恒不变式 = 逐 rid 逐链
        # borrow ≤ rid 名下该链路占用（_owner_rid_slots_on_link）——
        # 创建/转移后 O(1) 增量断言（_assert_borrow_within_occupancy），
        # release_merge sweep 前残留 = RuntimeError fail-closed。
        self._quota_epoch = 0
        # AIMD（C10）：链路峰值速率（扩张上界派生用）/ 逐链路遥测
        # 状态 / 遥测钟 / 流寿命 EWMA / 流准入时标（配对生成寿命样本）。
        self._b_link = float(noc_link_bytes_per_ns)
        self._aimd_states: dict = {}
        self._telemetry_now_ns: Optional[int] = None
        # K7（P2-7）：observe_telemetry 调用序号（缺测段"不进不出"
        # 判据——链路 last_seq 与当前序号衔接才算连续采样）。
        self._telemetry_seq: int = 0
        self._lifetime_ewma = FlowLifetimeEwma()
        self._flow_admit_ns: dict = {}

    # ------------------------------------------------------------ 只读 --
    @property
    def mode(self) -> str:
        return self._mode

    @property
    def rho_eff(self) -> float:
        return self._rho_eff

    @property
    def q_init(self) -> int:
        return self._q_init

    @property
    def n_bulk(self) -> int:
        return self._n_bulk

    @property
    def delta_adm_ns(self) -> int:
        return self._delta_adm_ns

    def quota_retry_key(self) -> tuple:
        """配额重试代数（C11 对接合同；经 :func:`extend_retry_key` 并入
        SH:1775-1802 的 epoch 键重试门）。"""
        return (self._quota_epoch,)

    @property
    def flow_lifetime_ewma_ns(self) -> Optional[float]:
        """EWMA(流寿命)（T_expand 的服务因子；None = 冷启动）。"""
        return self._lifetime_ewma.value_ns

    @property
    def flow_lifetime_samples(self) -> int:
        """已进入 EWMA 的正寿命样本数。"""
        return self._lifetime_ewma.samples

    @property
    def t_expand_ns(self) -> Optional[float]:
        """T_expand = k × EWMA(流寿命)（k=10，D5）；冷启动 = None。"""
        value = self._lifetime_ewma.value_ns
        return None if value is None else AIMD_EXPAND_K * value

    def aimd_link_state(self, link_id: Hashable) -> dict:
        """逐链路 AIMD 遥测状态披露（测试/日志位；未见遥测 = 空基线）。"""
        state = self._aimd_states.get(link_id)
        if state is None:
            return {"last_ns": None, "last_signal": None,
                    "last_seq": None, "quiet_ns": 0}
        return {
            "last_ns": state.last_ns,
            "last_signal": state.last_signal,
            "last_seq": state.last_seq,
            "quiet_ns": state.quiet_ns,
        }

    def link_quota(self, link_id: Hashable) -> int:
        return self._link_quota.get(link_id, self._q_init)

    def link_occupancy(self, link_id: Hashable) -> int:
        return sum(self._link_occ.get(link_id, {}).values())

    def link_reserved(self, link_id: Hashable) -> int:
        return sum(self._link_res.get(link_id, {}).values())

    def link_remaining(self, link_id: Hashable) -> int:
        """链路信用余量 = Q − 占用 − 预留（aimd 收缩后可为负：在册流
        不逐出、只封新准入，直到排空恢复正余量）。"""
        return (self.link_quota(link_id) - self.link_occupancy(link_id)
                - self.link_reserved(link_id))

    def _owner_rid_slots_on_link(self, link: Hashable, rid: str) -> int:
        """N1：rid 名下（owner == rid 或 rid# 前缀）流在该链路的占用
        槽位——同事务借槽的借主侧计量。"""
        prefix = rid + "#"
        return sum(
            slots for owner, slots in
            self._link_occ.get(link, {}).items()
            if owner == rid or owner.startswith(prefix))

    def _assert_borrow_within_occupancy(
            self, rid: str, link: Hashable) -> None:
        """O10②（2026-09-23 终轮深挖）：借槽账本守恒不变式——rid 名下
        借槽 ≤ rid 名下该链路占用槽位（借走的槽必须由本 rid 读流
        occupancy 实际承载）。创建（reserve_merge 落账）与转移
        （release_flow occ→res）后逐链增量校验（O(1)，非每操作全量
        扫）；转移取 min 的既有逻辑保证该式，断言把它从隐式不变式钉成
        显式 fail-closed——未来编辑写坏账账时在此 raise 而非静默漂移。
        """
        borrowed = self._merge_borrow.get(rid, {}).get(link, 0)
        occupied = self._owner_rid_slots_on_link(link, rid)
        if borrowed > occupied:
            raise RuntimeError(
                f"merge borrow ledger invariant violated: rid={rid!r} "
                f"link={link!r} borrowed={borrowed} > rid occupancy="
                f"{occupied} (borrow must be carried by rid flow "
                "occupancy; ledger desynchronized, fail-closed)")

    def port_enrolled(self, port_id: Hashable) -> int:
        return sum(count for count, _ in
                   self._port_enroll.get(port_id, {}).values())

    def port_enrolled_by_class(self, port_id: Hashable) -> dict:
        counts = {cls: 0 for cls in FLOW_CLASSES}
        for count, cls in self._port_enroll.get(port_id, {}).values():
            counts[cls] = counts.get(cls, 0) + count
        return counts

    def bulk_used(self, port_id: Hashable) -> int:
        return sum(self._port_bulk.get(port_id, {}).values())

    def bulk_remaining(self, port_id: Hashable) -> int:
        return self._n_bulk - self.bulk_used(port_id)

    def port_parity_admits(
        self,
        port_id: Hashable,
        r_hat_kv_bytes_per_ns: float,
        u_port_base: int = 0,
    ) -> bool:
        """本端口当前平价门裁决（u_port = 外部活跃 decode + 在册流）。"""
        return port_parity_admits(
            self._b_hbm,
            u_port_base + self.port_enrolled(port_id),
            r_hat_kv_bytes_per_ns,
        )

    def port_parity_headroom(
        self,
        port_id: Hashable,
        r_hat_kv_bytes_per_ns: float,
        u_port_base: int = 0,
    ) -> int:
        """本端口平价门余量（实时流计；port_snapshot 披露位）。"""
        return port_parity_headroom(
            self._b_hbm,
            u_port_base + self.port_enrolled(port_id),
            r_hat_kv_bytes_per_ns,
        )

    def snapshot(self) -> dict:
        """决策日志披露快照（C5 ``port_snapshot`` 的数据源，C11 接线）。"""
        links = {
            repr(link): {
                "quota": self.link_quota(link),
                "occupancy": self.link_occupancy(link),
                "reserved": self.link_reserved(link),
                "remaining": self.link_remaining(link),
                "occupancy_owners": dict(owners),
            }
            for link, owners in self._link_occ.items()
        }
        for link, owners in self._link_res.items():
            links.setdefault(repr(link), {
                "quota": self.link_quota(link),
                "occupancy": self.link_occupancy(link),
                "reserved": self.link_reserved(link),
                "remaining": self.link_remaining(link),
                "occupancy_owners": {},
            })["reserved"] = self.link_reserved(link)
        ports = {
            repr(port): {
                "enrolled_by_class": self.port_enrolled_by_class(port),
                "enrolled_total": self.port_enrolled(port),
                "bulk_used": self.bulk_used(port),
                "n_bulk": self._n_bulk,
                "bulk_remaining": self.bulk_remaining(port),
            }
            for port in set(self._port_enroll) | set(self._port_bulk)
        }
        return {
            "mode": self._mode,
            "rho_eff": self._rho_eff,
            "q_init": self._q_init,
            "n_bulk": self._n_bulk,
            "delta_adm_ns": self._delta_adm_ns,
            "quota_epoch": self._quota_epoch,
            "flow_lifetime_ewma_ns": self._lifetime_ewma.value_ns,
            "flow_lifetime_samples": self._lifetime_ewma.samples,
            "t_expand_ns": self.t_expand_ns,
            # N1：同事务借槽披露（rid -> {link: 槽数}——槽位由本 rid
            # 读流 occupancy 承载、未入 reserved 的部分）。
            "merge_borrowed": {
                rid: {repr(link): count for link, count in borrow.items()}
                for rid, borrow in self._merge_borrow.items()},
            "links": links,
            "ports": ports,
        }

    # ------------------------------------------------------------ 准入 --
    def admit_flow(
        self,
        *,
        owner: str,
        flow_class: str,
        links: Sequence[Hashable] = (),
        port_id: Optional[Hashable] = None,
        r_hat_kv_bytes_per_ns: Optional[float] = None,
        u_port_base: int = 0,
        slots: int = 1,
        now_ns: Optional[int] = None,
    ) -> QuotaVerdict:
        """动作级流准入（链路 ∪ 端口原子判定；任一判据不满足 = 整体
        不借、零部分登记）。

        * ``owner``：与 SH owner 约定同构的流标识（如 ``rid#decode#{j}``、
          ``rid#readplan``）；同一 owner 在册期间重复 admit = fail-closed
          （幻影双占用，配对纪律）。
        * ``links``：本流占用的有向链路序列（重复 = 多槽）。
        * ``port_id``：端点 HBM 端口（实时/弹性/一次性均入 u_port 在册）。
        * ``r_hat_kv_bytes_per_ns``：实时流必填（平价门分母来源，负载
          视图派生、因果）；弹性/一次性流不套平价门、忽略该参数。
        * ``u_port_base``：外部活跃 decode 消费流数（F4 负载视图口径，
          只进平价门分母、不进并发计数）。
        * ``slots``：占用槽位数（缺省 1）。
        * ``now_ns``：可选事件钟时标（AIMD 计时通道）——与
          :meth:`release_flow` 的时标配对生成流寿命样本喂
          :class:`FlowLifetimeEwma`；缺省 None = 不参与计时。
        """
        if flow_class not in FLOW_CLASSES:
            raise LinkQuotaError(
                f"flow_class must be one of {FLOW_CLASSES}, got "
                f"{flow_class!r}")
        if not isinstance(owner, str) or not owner:
            raise LinkQuotaError(
                f"owner must be a non-empty string, got {owner!r}")
        if owner in self._flows:
            raise LinkQuotaError(
                f"owner {owner!r} is already enrolled (phantom double "
                "occupancy; release before re-admitting)")
        if slots <= 0:
            raise LinkQuotaError(f"slots must be >= 1, got {slots!r}")
        if u_port_base < 0:
            raise LinkQuotaError(
                f"u_port_base must be non-negative, got {u_port_base!r}")
        if now_ns is not None:
            if isinstance(now_ns, bool) or not isinstance(now_ns, int):
                raise LinkQuotaError(
                    f"now_ns must be an integer, got {now_ns!r}")
            if now_ns < 0:
                raise LinkQuotaError(
                    f"now_ns must be non-negative, got {now_ns!r}")
        if not links and port_id is None:
            raise LinkQuotaError(
                "admit_flow requires at least one link or a port "
                f"(owner={owner!r})")
        if flow_class == FLOW_REALTIME and r_hat_kv_bytes_per_ns is None:
            raise LinkQuotaError(
                f"realtime flow {owner!r} requires r_hat_kv_bytes_per_ns "
                "(parity gate input, derived from the load view)")

        gated = self._mode != QUOTA_OFF
        link_links = tuple(links)

        # ---- 链路门（三类同判据：占用 + 预留 + 新增 <= Q；off 旁路）----
        if gated and link_links:
            demand: dict = {}
            for link in link_links:
                demand[link] = demand.get(link, 0) + slots
            for link, need in demand.items():
                quota = self.link_quota(link)
                remaining = quota - self.link_occupancy(link) - (
                    self.link_reserved(link))
                if remaining < need:
                    return QuotaVerdict(
                        admitted=False,
                        flow_class=flow_class,
                        resource_kind="link",
                        resource_id=link,
                        remaining=remaining,
                        wait_reason=WAIT_QUOTA_LINK,
                        inapplicable_reason=(
                            f"quota_link: link={link!r} remaining="
                            f"{remaining} of Q={quota}, need={need} "
                            f"(occupancy={self.link_occupancy(link)}, "
                            f"reserved={self.link_reserved(link)})"),
                    )

        # ---- 端口门（按类分判；off 旁路）----
        if gated and port_id is not None:
            if flow_class == FLOW_REALTIME:
                u_port = u_port_base + self.port_enrolled(port_id)
                rate = self._b_hbm / (u_port + slots)
                if rate < r_hat_kv_bytes_per_ns:
                    return QuotaVerdict(
                        admitted=False,
                        flow_class=flow_class,
                        resource_kind="port",
                        resource_id=port_id,
                        remaining=0,
                        wait_reason=WAIT_QUOTA_PORT,
                        inapplicable_reason=(
                            f"quota_port: port={port_id!r} parity "
                            f"B_HBM/(u_port+{slots})={rate:.6g} < "
                            f"r_hat_kv={r_hat_kv_bytes_per_ns:.6g} "
                            f"(u_port_base={u_port_base}, enrolled="
                            f"{self.port_enrolled(port_id)}; busy port "
                            "admits nothing, no max(1,.) floor)"),
                    )
            elif flow_class == FLOW_ELASTIC:
                enrolled = self.port_enrolled(port_id)
                if enrolled + slots > self._q_init:
                    return QuotaVerdict(
                        admitted=False,
                        flow_class=flow_class,
                        resource_kind="port",
                        resource_id=port_id,
                        remaining=self._q_init - enrolled,
                        wait_reason=WAIT_QUOTA_PORT,
                        inapplicable_reason=(
                            f"quota_port: port={port_id!r} concurrency "
                            f"{enrolled}+{slots} > cap Q_init="
                            f"{self._q_init} (elastic class enters "
                            "u_port, parity gate not applied)"),
                    )
            # oneshot：不设端口门（全额计价；在册计入 u_port、收紧后续
            # 实时流的平价门与弹性并发计数）。

        # ---- 借（原子：全部门过才登记）----
        self._flows[owner] = _FlowEnrollment(
            owner=owner, flow_class=flow_class, links=link_links,
            port_id=port_id, slots=slots)
        if now_ns is not None:
            self._flow_admit_ns[owner] = now_ns
        for link in link_links:
            self._link_occ.setdefault(link, {})[owner] = (
                self._link_occ.setdefault(link, {}).get(owner, 0) + slots)
        if port_id is not None:
            record = self._port_enroll.setdefault(port_id, {})
            count, cls = record.get(owner, (0, flow_class))
            record[owner] = (count + slots, flow_class)
        return QuotaVerdict(
            admitted=True,
            flow_class=flow_class,
            resource_kind=None,
            resource_id=None,
            remaining=None,
            wait_reason=None,
            inapplicable_reason="",
        )

    def release_flow(
        self, owner: str, now_ns: Optional[int] = None
    ) -> None:
        """流 settle：还（配对纪律——未知 owner = 漏释放/双释放，
        fail-closed）。释放 bump 配额代数（deferred 重试门重开）。

        ``now_ns``（可选，AIMD 计时通道）：与准入时标配对生成流寿命
        样本喂 :class:`FlowLifetimeEwma`（正寿命才计；任一侧无时标 =
        无样本；release 早于 admit = fail-closed，先于任何账本变更
        校验）。缺省 None = 不参与计时，簿记与代数语义不变。
        """
        if now_ns is not None:
            if isinstance(now_ns, bool) or not isinstance(now_ns, int):
                raise LinkQuotaError(
                    f"now_ns must be an integer, got {now_ns!r}")
            if now_ns < 0:
                raise LinkQuotaError(
                    f"now_ns must be non-negative, got {now_ns!r}")
        enrollment = self._flows.get(owner)
        if enrollment is None:
            raise LinkQuotaError(
                f"release_flow: owner {owner!r} is not enrolled "
                "(double release or leak; pairing violated)")
        admit_ns = self._flow_admit_ns.get(owner)
        if (now_ns is not None and admit_ns is not None
                and now_ns - admit_ns < 0):
            raise LinkQuotaError(
                f"release_flow clock moved backwards for {owner!r}: "
                f"now_ns={now_ns} < admit_ns={admit_ns}")
        self._flows.pop(owner)
        released_by_link: dict = {}
        for link in enrollment.links:
            owners = self._link_occ.get(link)
            if owners is None or owner not in owners:  # pragma: no cover
                raise LinkQuotaError(
                    f"link ledger desynchronized at {link!r} for "
                    f"{owner!r}")
            owners[owner] -= enrollment.slots
            released_by_link[link] = (
                released_by_link.get(link, 0) + enrollment.slots)
            if owners[owner] <= 0:
                del owners[owner]
            if not owners:
                del self._link_occ[link]
        # N1：同事务借槽转移——读流 settle 后其槽位被 merge 预留接管
        # （occ→res，occ+res 总计不变 ⇒ 无新准入窗口打开；预留已撤则
        # 借记已在 _release_merge_side 消解、此处空转）。
        rid = owner.split("#", 1)[0] if "#" in owner else owner
        borrow = self._merge_borrow.get(rid)
        if borrow:
            merge_owner = MERGE_RESERVE_OWNER_TEMPLATE.format(rid=rid)
            for link, released in released_by_link.items():
                transfer = min(borrow.get(link, 0), released)
                if transfer > 0:
                    self._link_res.setdefault(link, {})[merge_owner] = (
                        self._link_res.setdefault(link, {}).get(
                            merge_owner, 0) + transfer)
                    borrow[link] -= transfer
                    if borrow[link] <= 0:
                        del borrow[link]
            # O10②：转移后逐链守恒断言——剩余借槽 ≤ rid 名下剩余占用
            # （转移取 min 的既有逻辑保证该式；断言 = 隐式不变式钉成
            # 显式 fail-closed，覆盖仍挂账的链路，O(1)）。
            for link in borrow:
                self._assert_borrow_within_occupancy(rid, link)
            if not borrow:
                self._merge_borrow.pop(rid, None)
        if enrollment.port_id is not None:
            record = self._port_enroll.get(enrollment.port_id)
            if record is None or owner not in record:  # pragma: no cover
                raise LinkQuotaError(
                    f"port ledger desynchronized at "
                    f"{enrollment.port_id!r} for {owner!r}")
            count, _ = record[owner]
            count -= enrollment.slots
            if count <= 0:
                del record[owner]
            if not record:
                del self._port_enroll[enrollment.port_id]
        # 流寿命样本（AIMD 计时通道；两端时标配对 + 正寿命才计）。
        self._flow_admit_ns.pop(owner, None)
        if now_ns is not None and admit_ns is not None:
            lifetime_ns = now_ns - admit_ns
            if lifetime_ns > 0:
                self._lifetime_ewma.observe(lifetime_ns)
        self._quota_epoch += 1

    # ------------------------------------------------------ merge 预留 --
    def reserve_merge(
        self,
        rid: str,
        *,
        links_forward: Sequence[Hashable] = (),
        links_reverse: Sequence[Hashable] = (),
        port_forward: Optional[Hashable] = None,
        port_reverse: Optional[Hashable] = None,
    ) -> QuotaVerdict:
        """merge 义务预留（借）：双向各 1 槽 + 双候选胜者侧各 1 bulk 名额。

        链路侧占 ``reservation``（非 occupancy——预留不发射流，但计入
        链路信用余量）；端口侧占 bulk 名额（封顶 ``N_bulk = Q_init``）。
        原子判定：任一资源不足 → 零登记返回 quota_deferred。owner 字符
        串 = ``rid#merge-reserve``（冻结）。量级分层（copy/recompute 轮
        零预留；remote-read 轮 ≈ min(两侧保留量)）是调用方策略（C11/
        C14），双向各 1 槽原语功能上已覆盖翻转分支。
        """
        if rid in self._merges:
            raise LinkQuotaError(
                f"merge reserve for {rid!r} already exists (duplicate "
                "reservation; pairing violated)")
        links_fwd = tuple(links_forward)
        links_rev = tuple(links_reverse)
        if not (links_fwd or links_rev or port_forward is not None
                or port_reverse is not None):
            raise LinkQuotaError(
                f"reserve_merge for {rid!r} requires at least one link "
                "or port")
        gated = self._mode != QUOTA_OFF

        demand: dict = {}
        for link in links_fwd:
            demand[link] = demand.get(link, 0) + 1
        for link in links_rev:
            demand[link] = demand.get(link, 0) + 1

        # N1：同事务借槽——本 rid 流已占用的链路槽位与预留复用（时序
        # 先后不并发）；借走部分不检 remaining、不入 reserved。
        borrow_plan: dict = {}
        for link, need in demand.items():
            borrow_plan[link] = (
                min(need, self._owner_rid_slots_on_link(link, rid))
                if gated else 0)

        if gated:
            for link, need in demand.items():
                quota = self.link_quota(link)
                remaining = (quota - self.link_occupancy(link)
                             - self.link_reserved(link))
                if remaining < need - borrow_plan[link]:
                    return QuotaVerdict(
                        admitted=False,
                        flow_class=FLOW_ONESHOT,
                        resource_kind="link",
                        resource_id=link,
                        remaining=remaining,
                        wait_reason=WAIT_QUOTA_LINK,
                        inapplicable_reason=(
                            f"quota_link: merge-reserve link={link!r} "
                            f"remaining={remaining} of Q={quota}, "
                            f"need={need}"),
                    )
            # K3（P1-⑤，2026-09-23 外部审计）：端口侧与链路侧同构——
            # 先聚合逐端口需求再统一裁决。原实现逐方向独立验
            # ``remaining < 1``，port_forward == port_reverse 时实际需
            # 2 名额而两次检查各只验 1 槽（N_bulk 可被突破）。
            port_demand: dict = {}
            for _direction, port in (
                    (MERGE_DIRECTION_FORWARD, port_forward),
                    (MERGE_DIRECTION_REVERSE, port_reverse)):
                if port is not None:
                    port_demand[port] = port_demand.get(port, 0) + 1
            for port, need in port_demand.items():
                remaining = self.bulk_remaining(port)
                if remaining < need:
                    return QuotaVerdict(
                        admitted=False,
                        flow_class=FLOW_ONESHOT,
                        resource_kind="port",
                        resource_id=port,
                        remaining=remaining,
                        wait_reason=WAIT_QUOTA_PORT,
                        inapplicable_reason=(
                            f"quota_port: merge bulk-write slots at "
                            f"port={port!r} exhausted: used="
                            f"{self.bulk_used(port)} of "
                            f"N_bulk={self._n_bulk} (aggregated "
                            f"need={need} across forward/reverse winner "
                            "sides)"),
                    )

        reserve = _MergeReserve(
            rid=rid, links_forward=links_fwd, links_reverse=links_rev,
            port_forward=port_forward, port_reverse=port_reverse)
        owner = reserve.owner
        for link in links_fwd:
            self._link_res.setdefault(link, {})[owner] = (
                self._link_res.setdefault(link, {}).get(owner, 0) + 1)
        for link in links_rev:
            self._link_res.setdefault(link, {})[owner] = (
                self._link_res.setdefault(link, {}).get(owner, 0) + 1)
        # N1：借走部分不入 reserved（槽位由本 rid 读流 occupancy 承载，
        # 同事务时序复用）；fwd/rev 同链时合并 need 借槽只扣一次。
        borrow_active = {
            link: borrowed for link, borrowed in borrow_plan.items()
            if borrowed > 0}
        for link, borrowed in borrow_active.items():
            owners = self._link_res.setdefault(link, {})
            owners[owner] = owners.get(owner, 0) - borrowed
            if owners[owner] <= 0:
                del owners[owner]
            if not owners:
                del self._link_res[link]
        if borrow_active:
            existing = self._merge_borrow.setdefault(rid, {})
            for link, borrowed in borrow_active.items():
                existing[link] = existing.get(link, 0) + borrowed
            # O10②：创建后逐链守恒断言——合并借槽（含既有挂账叠加）
            # ≤ rid 名下该链路占用（min(need, occ) 的既有落账保证该式；
            # 断言 = 隐式不变式钉成显式 fail-closed，O(1)）。
            for link in borrow_active:
                self._assert_borrow_within_occupancy(rid, link)
        if port_forward is not None:
            self._port_bulk.setdefault(port_forward, {})[owner] = (
                self._port_bulk.setdefault(port_forward, {}).get(owner, 0)
                + 1)
        if port_reverse is not None:
            self._port_bulk.setdefault(port_reverse, {})[owner] = (
                self._port_bulk.setdefault(port_reverse, {}).get(owner, 0)
                + 1)
        self._merges[rid] = reserve
        return QuotaVerdict(
            admitted=True, flow_class=FLOW_ONESHOT, resource_kind=None,
            resource_id=None, remaining=None, wait_reason=None,
            inapplicable_reason="")

    def adjudicate_merge_direction(self, rid: str, winner: str) -> None:
        """service_done（计算/响应完成）时刻方向裁决：释放**败者侧**
        （败者方向链路槽 + 败者候选端口 bulk 名额）。胜者侧保持到
        ``_on_merge_done``（:meth:`release_merge`）。"""
        if winner not in MERGE_DIRECTIONS:
            raise LinkQuotaError(
                f"winner must be one of {MERGE_DIRECTIONS}, got "
                f"{winner!r}")
        reserve = self._merges.get(rid)
        if reserve is None:
            raise LinkQuotaError(
                f"adjudicate_merge_direction: no merge reserve for "
                f"{rid!r} (unknown or already fully released)")
        if reserve.adjudicated is not None:
            raise LinkQuotaError(
                f"merge reserve for {rid!r} already adjudicated to "
                f"{reserve.adjudicated!r} (double adjudication)")
        loser_links, loser_port = (
            (reserve.links_reverse, reserve.port_reverse)
            if winner == MERGE_DIRECTION_FORWARD
            else (reserve.links_forward, reserve.port_forward))
        self._release_merge_side(reserve, loser_links, loser_port)
        reserve.adjudicated = winner
        self._quota_epoch += 1

    def release_merge(self, rid: str) -> None:
        """merge_done（SH:2099-2112 ``_on_merge_done``）释放：裁决后 =
        胜者侧；未裁决即整体释放 = 双侧撤销（remote 轮取消/翻转前撤回
        路径）。实际合并流的链路占用与端口在册由调用方按 oneshot 类
        ``admit_flow`` 另行登记（预留 → 实占的转换在 SH 侧完成）。"""
        reserve = self._merges.pop(rid, None)
        if reserve is None:
            raise LinkQuotaError(
                f"release_merge: no merge reserve for {rid!r} (double "
                "release or unknown reservation)")
        if reserve.adjudicated == MERGE_DIRECTION_FORWARD:
            self._release_merge_side(
                reserve, reserve.links_forward, reserve.port_forward)
        elif reserve.adjudicated == MERGE_DIRECTION_REVERSE:
            self._release_merge_side(
                reserve, reserve.links_reverse, reserve.port_reverse)
        else:
            self._release_merge_side(
                reserve, reserve.links_forward, reserve.port_forward)
            self._release_merge_side(
                reserve, reserve.links_reverse, reserve.port_reverse)
        # O10②：sweep 守恒断言——各侧逐链消解/转移后借槽应已清空，
        # 残留 = 账本脱钩，fail-closed（替代 N1 的静默弃置：删除前
        # 不静默，写坏账账的未来编辑在此 raise 而非被掩盖）。
        if self._merge_borrow.get(rid):
            raise RuntimeError(
                f"merge borrow ledger desynchronized at release_merge: "
                f"rid={rid!r} residue={self._merge_borrow[rid]!r} after "
                "all reservation sides released (borrow must be empty "
                "once the reservation is fully released; fail-closed)")
        self._merge_borrow.pop(rid, None)
        self._quota_epoch += 1

    def _release_merge_side(self, reserve: _MergeReserve, links, port) -> None:
        owner = reserve.owner
        for link in links:
            owners = self._link_res.get(link)
            current = owners.get(owner, 0) if owners else 0
            if current > 0:
                owners[owner] = current - 1
                if owners[owner] <= 0:
                    del owners[owner]
                if not owners:
                    del self._link_res[link]
            else:
                # N1：该侧槽位为同事务借槽（未转移）——预留撤销时借记
                # 就地消解（读流槽位回归自身 occupancy 语义，无净变化）。
                borrow = self._merge_borrow.get(reserve.rid)
                if borrow and borrow.get(link, 0) > 0:
                    borrow[link] -= 1
                    if borrow[link] <= 0:
                        del borrow[link]
                    if not borrow:
                        self._merge_borrow.pop(reserve.rid, None)
                    continue
                raise LinkQuotaError(
                    f"merge reservation ledger desynchronized at "
                    f"{link!r} for {owner!r}")
        if port is not None:
            owners = self._port_bulk.get(port)
            if owners is None or owners.get(owner, 0) <= 0:  # pragma: no cover
                raise LinkQuotaError(
                    f"merge bulk ledger desynchronized at {port!r} for "
                    f"{owner!r}")
            owners[owner] -= 1
            if owners[owner] <= 0:
                del owners[owner]
            if not owners:
                del self._port_bulk[port]

    # ------------------------------------------------------------ AIMD --
    def observe_telemetry(
        self,
        now_ns: int,
        link_telemetry: Mapping[Hashable, float],
        r_hat_kv_bytes_per_ns: float,
        *,
        allow_expansion: bool = True,
    ) -> dict:
        """AIMD 遥测消费（冻结契约 ``{link_id: 实测有效速率}``；仅
        ``mode=aimd``，static/off fail-closed——与 :meth:`set_link_quota`
        同律）。

        value = 该链路在册流的实测**每流**有效速率（bytes/ns），直接
        与 ``r_hat_kv_bytes_per_ns``（负载视图当前值，因果）比较。逐
        链路判据（:func:`aimd_band_signal`，D5 胀缩阈值带）：

        * ``shrink``（rate < r_KV）：收缩事件——MD 减半（Q 已在 1 时
          记事件不调整）；streak 清零。
        * ``hold``（死区 [r_KV, 1.2·r_KV)）：不动作，streak 清零
          （死区停留打断"连续舒适"——两支控制律无公共触发态）。
        * ``comfort``（rate >= 1.2·r_KV）：streak += dt（与上次采样
          的间隔）；累计 >= ``T_expand = k × EWMA(流寿命)`` 时扩张
          ``Q ← min(Q+1, floor(B_link/r_KV))``（扩张上界 = 收缩等值
          线本身）并清零 streak。冷启动（无寿命样本）= 扩张抑制，
          streak 不累计。

        ``allow_expansion``（O6②，关键字参数，缺省 True = 现状零行为
        变更）：False = **扩张冻结窗口**——additive-increase 扩张与
        expand_capped 均不执行，并清除此前 quiet_ns。空 telemetry 的
        冻结调用也清理所有链路进度；紧邻冻结样本的首个解冻样本不计
        冻结区间 dt。冻结期间收缩、保持、comfort/shrink 判定、遥测
        钟/序号簿记、EWMA（经 admit/release 时标通道）全部照常。
        O6 口径：r̂ 代表值回退窗口（active_decode 瞬空、SH 侧
        ``_quota_r_hat_kv_bytes_per_ns`` 走代表值 ctx=1）由调用方传
        False——空窗不扩张、decode 回归不瀑布（解冻后须从零连续累计
        满 T_expand 才扩张）。

        遥测钟非降（倒退 fail-closed）；同 tick 多次采样 dt=0 合法。
        返回逐链路动作披露（signal/action/dt_ns/quiet_ns/quota_before/
        quota_after + t_expand_ns/allow_expansion），供决策日志与
        稳定性审计消费。字典中不出现的链路 = 无测量（无在册流），
        状态冻结不动。
        """
        if self._mode != QUOTA_AIMD:
            raise LinkQuotaError(
                f"observe_telemetry is only legal in {QUOTA_AIMD!r} mode "
                f"(static keeps Q = Q_init; got mode={self._mode!r})")
        if isinstance(now_ns, bool) or not isinstance(now_ns, int):
            raise LinkQuotaError(f"now_ns must be an integer, got {now_ns!r}")
        if now_ns < 0:
            raise LinkQuotaError(
                f"now_ns must be non-negative, got {now_ns!r}")
        if (self._telemetry_now_ns is not None
                and now_ns < self._telemetry_now_ns):
            raise LinkQuotaError(
                "telemetry clock must be non-decreasing: now_ns="
                f"{now_ns} < last={self._telemetry_now_ns}")
        if (not (r_hat_kv_bytes_per_ns > 0)
                or not math.isfinite(r_hat_kv_bytes_per_ns)):
            raise LinkQuotaError(
                "r_hat_kv_bytes_per_ns must be a positive finite rate, "
                f"got {r_hat_kv_bytes_per_ns!r}")
        for link, rate in link_telemetry.items():
            if not (rate > 0) or not math.isfinite(rate):
                raise LinkQuotaError(
                    f"link_telemetry[{link!r}] must be a positive finite "
                    f"rate, got {rate!r}")
        if not isinstance(allow_expansion, bool):
            raise LinkQuotaError(
                f"allow_expansion must be a bool, got "
                f"{allow_expansion!r}")
        t_expand = self.t_expand_ns
        # K7（P2-7）：本调用序号——链路 last_seq 衔接（== seq-1）才算
        # 连续采样；普通缺测段的 dt 不计、quiet_ns 保留。显式冻结与普通
        # 缺测不同：无论字典是否为空，先清除所有已有 quiet 进度。
        seq = self._telemetry_seq + 1
        links_out: dict = {}
        if not allow_expansion:
            for state in self._aimd_states.values():
                state.quiet_ns = 0
        for link, rate in link_telemetry.items():
            signal = aimd_band_signal(rate, r_hat_kv_bytes_per_ns)
            state = self._aimd_states.setdefault(link, _AimdLinkState())
            dt_ns = 0 if state.last_ns is None else now_ns - state.last_ns
            contiguous = state.last_seq is not None and state.last_seq == seq - 1
            quota_before = self.link_quota(link)
            quota_after = quota_before
            if signal == AIMD_SIGNAL_SHRINK:
                state.quiet_ns = 0
                target = aimd_shrink_quota(quota_before)
                if target < quota_before:
                    self.set_link_quota(link, target)
                    quota_after = target
                    action = AIMD_ACTION_SHRINK
                else:
                    action = AIMD_ACTION_SHRINK_FLOOR
            elif signal == AIMD_SIGNAL_HOLD:
                state.quiet_ns = 0
                action = AIMD_ACTION_HOLD
            else:                                   # AIMD_SIGNAL_COMFORT
                if t_expand is None:
                    action = AIMD_ACTION_COMFORT_COLD_START
                elif not allow_expansion:
                    # O6②：显式冻结清除既有进度；该样本更新 last_*，
                    # 供解冻后的首样本跳过冻结区间 dt。
                    state.quiet_ns = 0
                    action = AIMD_ACTION_COMFORT_FROZEN
                else:
                    # K7（P2-7）：仅连续采样计 dt——缺测后首个样本
                    # dt 含整段缺测时长，计入即"单样本触发扩张"
                    # （T_expand 阻尼被架空）。
                    fresh_interval = (
                        contiguous and state.last_expansion_allowed is not False
                    )
                    state.quiet_ns += dt_ns if fresh_interval else 0
                    if state.quiet_ns >= t_expand:
                        ceiling = aimd_expand_ceiling(
                            self._b_link, r_hat_kv_bytes_per_ns)
                        target = min(quota_before + 1, ceiling)
                        if target > quota_before:
                            self.set_link_quota(link, target)
                            quota_after = target
                            action = AIMD_ACTION_EXPAND
                        else:
                            action = AIMD_ACTION_EXPAND_CAPPED
                        state.quiet_ns = 0
                    else:
                        action = AIMD_ACTION_COMFORT
            state.last_ns = now_ns
            state.last_signal = signal
            state.last_seq = seq
            state.last_expansion_allowed = allow_expansion
            links_out[repr(link)] = {
                "signal": signal,
                "action": action,
                "dt_ns": dt_ns,
                "contiguous": contiguous,
                "quiet_ns": state.quiet_ns,
                "quota_before": quota_before,
                "quota_after": quota_after,
            }
        self._telemetry_seq = seq
        self._telemetry_now_ns = now_ns
        return {
            "now_ns": now_ns,
            "r_hat_kv_bytes_per_ns": r_hat_kv_bytes_per_ns,
            "t_expand_ns": t_expand,
            "allow_expansion": allow_expansion,
            "links": links_out,
        }

    def set_link_quota(self, link_id: Hashable, quota: int) -> None:
        """链路 Q 调整原语（**仅 aimd 模式**；AIMD 控制律
        :meth:`observe_telemetry` 的收缩/扩张均经本原语落地）。

        收缩允许低于当前占用+预留（在册流不逐出 = grandfathered，余量
        为负期间封新准入，直到排空）；下限 1 = 防结构性死锁（F2 同源）。
        调整 bump 配额代数（deferred 重试门重开——配额变化改变可准入性）。
        K7（P2-8，2026-09-23 外部审计）：同值调用**零 bump**——代数
        bump 的语义是"可准入性变化"（非释放侧事件零 bump 纪律），同值
        设置不改变任何判据结果，bump 只制造 deferred 虚假唤醒。
        """
        if self._mode != QUOTA_AIMD:
            raise LinkQuotaError(
                f"set_link_quota is only legal in {QUOTA_AIMD!r} mode "
                f"(static keeps Q = Q_init; got mode={self._mode!r})")
        if not isinstance(quota, int) or isinstance(quota, bool):
            raise LinkQuotaError(
                f"quota must be an integer, got {quota!r}")
        new_quota = max(1, quota)
        if self._link_quota.get(link_id) != new_quota:
            self._link_quota[link_id] = new_quota
            self._quota_epoch += 1
