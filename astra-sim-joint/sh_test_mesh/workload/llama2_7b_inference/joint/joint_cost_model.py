"""joint_cost_model.py -- J：无 oracle 的 (instance × action) 完成时间预测。

设计依据：《三机制联合策略_template仓库设计方案》§6、§9.2（实验 joint
选择性移植的六项缺口修正）与实验总纲 §13.2。

统一预测目标：``cost(instance, action)`` 预测到同一 ``merge_done``
边界的完成时间——包括目标等待、历史准备、执行期增长、remote 路径、
compute_done 后的 home 释放与合并（F11：预测终点 = merge_done，四动作
同终点；service_done（响应完成）作为服务指标从事件侧分别报告——本
docstring 原先的 ``service_done`` 字样为旧版合同术语残留，C4-G2 术语
rider 统一为 merge_done 终点表述，语义零变更，PROVENANCE §15.3 命名
映射勘误的移交落地）。重叠工作按依赖关键路径处理，不机械相加，不把
同一拥塞重复计费。

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

状态（§5.6 表，R15 后 2026-09-14；A8' 修订 2026-09-22）：本模块按
"已实现（解析近似 + 在线因子）"交付，且在线反馈通道**已接线**——
``LinkFlowRegistry`` 由调度器在五类传输发射时逐 shard 全路径登记、
完成事件注销（F-B 聚合除数）；``ServiceFactors`` 三观测入口挂列车
完成事件（P3 α 公式）——transfer 因子的样本源 = SH 侧链路遥测窗口
（A8' 接线：``_ingest_link_telemetry`` 逐窗口调
``observe_transfer_from_link_window``，列车核销通道无纯传输段可因果
分离故不经该通道；仅遥测开启路径有样本，缺省关臂 updates=0 披露）；
池端口除数注入（含 E 内核 r_j，R15-3）。``collective_coverage`` 仍
由调用方按遥测覆盖披露。

C1（2026-09-22，WP1a）：NoC 计价四点（copy 前缀腿 / remote-read
读流与 credit 切块 / merge 双腿）改为**逐 shard 三腿 min**（F1 冻
结）——noc_stream = bytes_s/(B_link/divisor_multi(候选路径并集))、
home_read = bytes_s/(B_HBM/(u_home+1))、exec_write 仅 copy/staging；
wall = max_s(startup + d2d·hops(path_s) + max(腿))，TP 并行不再按
"聚合字节 ÷ 单链路率"串行放大；``local_hbm_bytes_per_ns`` 自此正式
消费（端点腿；E12）。D1 读放大因果口径：decode 每步远读基数 = 仍
位于远端的基础历史，执行端生成的 input/decode 增量本地读取、不计
作远读；prefill 按实际消费遍数计（设计文档 §1.4）。

保真边界披露（F3/C1）：**池路径 NoC 腿未计价**——``_pool_transfer_ns``
五消费点（stay 池恢复 / copy 池恢复 / remote 混合后缀 / merge 无 /
驱逐写回）按池端口口径计价，池↔实例间链路段不叠加链路腿。

C2（2026-09-22，WP1b）：``hbm_port_registry``（HbmPortFlowRegistry，
WL/joint/hbm_port_flow_registry.py）自本卡起供给端点腿除数 u_home/
u_exec（F4：活跃 decode 消费流 + 在册传输/远读流；未注入时 u=0，端点
腿独占——离线/单测口径）。A4' 补价 rider（PROVENANCE §20.1，C14 G1
裁定 (a)）：kind="read" 增执行端 HBM 写腿——与 copy/staging 第三腿同
式同源（镜像执行侧 COMM_WRITE 服务类），remote-read 计价自此三腿；
仅增带宽/服务争用腿，不改 F14 容量半边（credit 在途字节仍不占执行端
HBM 容量、暂存归还仍无操作）。E4 去重（A1 勘误）：breakdown 的
contention_divisor 复用主计价腿已算得的并集除数，estimate_action 不
再对 candidate_paths 二次调用 divisor_multi（stay/recompute 无计价腿
可复用时维持单次调用）。u_port 分解披露走决策日志 notes 通道（C5 冻
结 schema 不动；port_snapshot 正式字段位归 C11 步骤 6）。

C7（2026-09-22，WP2，仅 JCM 半）：C++ 侧逐链路窗口遥测（C6
``link_telemetry[]``）的代价模型消费三件——(1) ``divisor_effective``：
逐链路有效除数 = ``max(divisor_registered, B_link / 实测有效速率)``
（``TelemetryLinkFlowView``；取 max 保守合并：注册表漏计时实测抬起、
多计时不放大——max 而非相加即设计文档 §4.1"自注册计划和遥测若反映
同一流量，必须去重"）；(2) ``collective_coverage`` 翻 True 条件求值
（``LinkFlowRegistry.evaluate_collective_coverage`` 纯函数：全程遥测
开启 ∧ 逐 epoch 窗口序列无空洞；置位/决策日志/manifest 接线在 SH 侧
C8）；(3) ``transfer_factor`` EWMA 转正（观测入口
``ServiceFactors.observe_transfer_from_link_window``：遥测链路窗口
actual=active_ns / base=served/名义速率，样本比 = 名义速率/实测有效
速率 = 该链路遥测有效除数，≥1 拥胀方向与因子族 actual/base 口径一
致）。冻结交接接口（C8 联合冻结，kwarg 名对齐）：SH/C8 以
``link_telemetry_rates = {link_id: 实测有效速率(B/ns)}`` 传入
``JointCostModel`` 构造；None/空 = 遥测未开启，注册表除数模型零漂
移。键两形：端点形 ``(src, dst)``/``"src->dst"`` 参与端点除数合并；
裸整型 LinkId（C8 ``_ingest_link_telemetry`` 的原样键）端点无关消
费（速率查询/transfer_factor 观测/披露计数），不参与端点除数合并
——LinkId→有向链路换算需拓扑知识，缺口与转换配方登记见
``TelemetryLinkFlowView`` docstring 与 C7 BLOCKED。

A8'（2026-09-22，§4.3 补遗，F1）：上件 (3) 的 SH 侧调用落地——
``_ingest_link_telemetry`` 逐遥测窗口调
``observe_transfer_from_link_window``（端点键换算复用
``_telemetry_endpoint_link_key``；名义速率 = noc_link_bytes_per_ns），
"三件已交付"自本卡起为全链真值（此前 SH 半零调用、transfer_factor
恒 1.0/updates 恒 0 的名实不符消除；行为面 = 仅遥测开启路径，缺省关
臂零扰动）。

规格书 §5（2026-09-25，remote-read 分阶段关键路径）：remote-read 候
选的关键路径合成改为 prefill/decode 两阶段——prefill 前缀读流
（home→exec）与后缀池恢复（pool→exec HBM）自同一准入 frontier 并行
分叉，prefill 计算按层段随数据到达推进（前缀段等首 credit、后缀段
等恢复，无"全量后缀恢复完成才开始 prefill"的全局 barrier）；decode
只对 home 前缀新发 credit 读、后缀在 exec HBM 就地复用不再恢复；
merge 段仍为 merge v2。合成式：prefill_stage = prefix_first_credit +
max(prefix_remaining_stream, suffix_restore, prefill_compute)、
decode_stage = decode_first_credit + max(decode_remaining_stream,
decode_compute)、total = target_wait + eviction_wait + prefill_stage
+ decode_stage + merge。旧 max(history_prep, eviction_wait) +
first_credit + max(remaining_stream, compute)（后缀恢复全量串在
prefill 计算之前）废除、无开关可选项。breakdown 冻结字段口径全部
保持（C5），阶段量经 notes 四键披露：remote_read_prefill_ns /
remote_read_decode_ns / suffix_restore_ns /
prefill_pipeline_overlap_ns。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Hashable, Mapping, Optional, Sequence

from joint.joint_config import remote_credit_block_size

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
    # A12'：亚名义样本钳位 1.0 的披露计数（仅 transfer_factor）。
    # A14'/H10 计数口径注记：计的是被钳的"同 tick 聚合更新"条数
    # （_record 的 Σactual/Σbase 汇总纪律 + observe_transfer_from_
    # link_window 的 base_ns 向下取整）——同 tick 内亚名义链路可被
    # 拥塞链路稀释为 ≥1 聚合比而不计数、真比值略 <1 的窗口可因 base
    # 截断偏小逃逸钳位；披露语义是"聚合更新钳位次数"而非"亚名义链
    # 路窗口数"（因子恒 ≥1 的不变量不受影响）。
    clamped: dict = field(default_factory=dict)
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

    def observe_transfer_from_link_window(
        self, *, served_bytes: int, active_ns: int,
        nominal_rate_bytes_per_ns: float, tick_ns: int,
    ) -> None:
        """C7（WP2）transfer_factor 转正：遥测链路窗口样本 → observe_transfer。

        观测源 = 遥测有效速率对名义速率之比（C6 ``link_telemetry[]`` 单
        元的 ``served_bytes``/``active_ns`` ÷ 配置链路峰值速率）。按
        ``_record`` 既有 actual/base 纪律落样：actual_ns = active_ns（纯
        服务段实测——链路活跃积分，无排队/merge 尾段混入），base_ns =
        served_bytes / 名义速率（同字节量在峰值速率下的基准时长）⇒ 样
        本比 = 名义速率 / 实测有效速率 = 该链路遥测有效除数（与
        ``divisor_effective`` 的 B_link/measured 同式同源，≥1 拥胀方向；
        ServiceFactors 因子族口径 = actual/base、1.0 冷启动"乐观可能"，
        本方向与之一致——亚名义样本（ratio<1）经 ``_apply_pending``
        钳位 1.0（A12'，§4.3 补遗）并以 clamped 计数披露）。同 tick
        多链路样本经 ``_record`` 时刻汇总纪律
        合并为一条更新（Σactual/Σbase）；零字节/零活跃/非有限样本由
        ``_record`` 既有拒绝计数披露（本入口不预滤，调用方逐链路如实
        喂入即可）。
        """
        if nominal_rate_bytes_per_ns <= 0 or not math.isfinite(
                nominal_rate_bytes_per_ns):
            # 名义速率非法 = 配置错误，fail-closed（非样本拒绝路径）。
            raise JointCostError(
                "nominal_rate_bytes_per_ns must be a positive finite rate, "
                f"got {nominal_rate_bytes_per_ns!r}")
        base_ns = int(served_bytes / nominal_rate_bytes_per_ns)
        self.observe_transfer(
            actual_ns=int(active_ns), base_ns=base_ns,
            service_ns=int(active_ns), tick_ns=int(tick_ns))

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
        if name == "transfer_factor" and ratio < 1.0:
            # A8'/A12'：transfer 样本方向冻结 = ≥1 拥胀——亚名义样本
            # （实测快于名义：active_ns 欠计/空闲混入）钳位 1.0 并计数
            # 披露；prefill/decode 因子不钳（历史口径允许 <1）。
            ratio = 1.0
            self.clamped[name] = self.clamped.get(name, 0) + 1
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
            "clamped": dict(self.clamped),
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

    C7（2026-09-22，WP2）：遥测消费三入口——``with_effective_rates``
    构造遥测叠加视图（divisor_effective 的 max 去重合并，见
    ``TelemetryLinkFlowView``）；``evaluate_collective_coverage`` 为
    ``collective_coverage`` 翻 True 条件的纯函数求值（置位接线在 SH
    侧 C8，本注册表的 ``collective_coverage`` 属性仍为调用方披露载
    体、缺省 False）。
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
    def registered_flows(self, link: tuple[int, int]) -> int:
        """该有向链路的已登记流数（N2：TelemetryLinkFlowView 合并口径
        的公共访问器——注册侧旧流并发计量，不含候选自身）。"""
        return self._flows.get(link, 0)

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
        self_counts = _path_union_links(paths)
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

    def leaked_owners(self) -> dict:
        """在册归属视图（漏释放审计：结算边界后仍非空 = 泄漏证据）。

        O10③（A19'(j)③，2026-09-23）：与 HbmPortFlowRegistry.
        leaked_owners 同款语义与返回形态——``{owner: tuple(在途流 id)}``
        有序 dict，空 = 干净。只统计仍实际在途（流 id 未注销）的流：
        ``release_flow`` 单独释放的流 id 在 ``_owner_flows`` 的残留
        不算泄漏（物理链路已注销；生产完成事件全走 ``release_owner``，
        该过滤为防御纵深）。
        """
        leaked = {}
        for owner, flow_ids in sorted(self._owner_flows.items()):
            live = tuple(
                flow_id for flow_id in flow_ids
                if flow_id in self._flow_links)
            if live:
                leaked[owner] = live
        return leaked

    # ---------------------------------------------------- 遥测（C7） --
    def with_effective_rates(
        self,
        effective_rates: Mapping[Hashable, float],
        *,
        link_capacity_bytes_per_ns: float,
        link_flow_counts: Optional[Mapping[Hashable, float]] = None,
    ) -> "TelemetryLinkFlowView":
        """C7：以 ``{link_id: 实测有效速率(bytes/ns)}`` 构造遥测叠加
        视图（冻结交接接口的 JCM 侧消费原语；JointCostModel.link_
        telemetry 经本方法接线）。返回视图的 divisor 族按 max 与本注
        册表合并——本注册表自身状态零变更（决策时点只读快照纪律）。

        M1（2026-09-23 验收审计）：``link_flow_counts`` = C++ 时间加权
        活跃流数（含 collective）——在场时遥测除数取**流数口径**
        （物理除数 = 链上流数；合计吞吐无法反推流数——满载链路
        合计≈容量与流数无关、下游瓶颈流会被 capacity/合计 误读为
        争用），缺席退 capacity/速率 旧口径（既有测试注入单流速率
        语义）。
        """
        return TelemetryLinkFlowView(
            self, effective_rates,
            link_capacity_bytes_per_ns=link_capacity_bytes_per_ns,
            link_flow_counts=link_flow_counts)

    @staticmethod
    def evaluate_collective_coverage(
        *,
        telemetry_enabled_whole_run: bool,
        window_bounds: Sequence[tuple[int, int]],
        run_start_ns: int = 0,
        run_end_ns: Optional[int] = None,
    ) -> tuple[bool, dict]:
        """C7：``collective_coverage`` 翻 True 条件求值（纯函数）。

        条件 = 该 run **全程遥测开启** ∧ **窗口序列无空洞**：

        * ``telemetry_enabled_whole_run``：调用方对 ``--link-telemetry``
          自 run 起持续供给的布尔汇总（中途开启/中途掉线 = False）；
        * 无空洞：``window_bounds`` 为**逐 delivery epoch** 的
          (start_ns, end_ns) 序列——含全闲 epoch（其 ``link_telemetry[]``
          为空数组但 epoch 存在，界值取该次 delivery 的 tick 元数据；
          只报非空数组的 epoch 会把全闲 epoch 误判为空洞，属调用方契
          约违背）。判据：首窗 start == ``run_start_ns``（缺省 0）、相
          邻窗 start_{i+1} == end_i、给 ``run_end_ns`` 时末窗 end ==
          run_end_ns；零长窗 [t, t)（同 tick 双 delivery）合法，负长度
          窗（end < start）计为空洞。

        字段翻转接线（置位本类 ``collective_coverage``、决策日志
        ``contention_coverage``、manifest 记录遥测完备性）均在 SH 侧
        C8；本方法只做条件求值。返回 ``(flag, 披露 dict)``，披露含空洞
        明细（前 8 条）与覆盖跨度，供 manifest 落盘。
        """
        bounds = [(int(start), int(end)) for start, end in window_bounds]
        holes: list = []
        if bounds:
            if bounds[0][0] != int(run_start_ns):
                holes.append(
                    {"index": 0, "kind": "start",
                     "expected_start_ns": int(run_start_ns),
                     "got_start_ns": bounds[0][0]})
            for index in range(1, len(bounds)):
                prev_end = bounds[index - 1][1]
                start, end = bounds[index]
                if end < start:
                    holes.append(
                        {"index": index, "kind": "negative_window",
                         "start_ns": start, "end_ns": end})
                if start != prev_end:
                    holes.append(
                        {"index": index, "kind": "gap",
                         "expected_start_ns": prev_end,
                         "got_start_ns": start})
            if bounds[0][1] < bounds[0][0]:
                holes.append(
                    {"index": 0, "kind": "negative_window",
                     "start_ns": bounds[0][0], "end_ns": bounds[0][1]})
            if run_end_ns is not None and bounds[-1][1] != int(run_end_ns):
                holes.append(
                    {"index": len(bounds) - 1, "kind": "end",
                     "expected_end_ns": int(run_end_ns),
                     "got_end_ns": bounds[-1][1]})
        covered_span_ns = (
            bounds[-1][1] - bounds[0][0] if bounds else 0)
        flag = bool(
            telemetry_enabled_whole_run and bounds and not holes)
        return flag, {
            "telemetry_enabled_whole_run": bool(telemetry_enabled_whole_run),
            "window_count": len(bounds),
            "run_start_ns": int(run_start_ns),
            "run_end_ns": (None if run_end_ns is None else int(run_end_ns)),
            "covered_span_ns": covered_span_ns,
            "holes": holes[:8],
            "hole_count": len(holes),
        }


def _path_union_links(
    paths: Sequence[Sequence[int]],
) -> dict[tuple[int, int], int]:
    """候选路径族的链路并集：``{(src, dst): 自身经过次数}``。

    F-B 口径（LinkFlowRegistry.divisor_multi 的既有循环原样抽出，零
    语义漂移）：短于 2 的路径无链路不计；共享链路按经过次数累加。
    """
    self_counts: dict[tuple[int, int], int] = {}
    for path in paths:
        if len(path) < 2:
            continue
        for index in range(len(path) - 1):
            key = (path[index], path[index + 1])
            self_counts[key] = self_counts.get(key, 0) + 1
    return self_counts


class TelemetryLinkFlowView:
    """LinkFlowRegistry 的遥测叠加只读视图（C7，WP2：divisor_effective）。

    逐链路有效除数（冻结公式，卡 C7 步骤 1）::

        divisor_effective(link) = max(
            divisor_registered(link),              # 注册表模型（含自身）
            link_capacity_Bps / measured_effective_rate(link))  # 遥测实测

    **取 max（保守）**：注册表漏计时实测抬起来，多计时不放大；max 而
    非相加 = 同一流量不重复计数（设计文档 §4.1"自注册计划和遥测若反
    映同一流量，必须去重"）。max 的层间可交换性使"逐链路合并后取瓶颈
    ≡ max(注册表瓶颈, 遥测下界)"——divisor/divisor_multi 据此以一次注
    册表求值 + 一次遥测下界扫描实现，无逻辑分叉。

    **窗口均值 ≠ 决策瞬时**（A18'(b) 登记补履行，O7④/A19'(g)④，
    2026-09-23 运行时代码落点）：遥测流数/速率为 C++ 周期采样窗口的
    时间加权均值，非决策时刻瞬时值；流进出频繁时均值系统性低于瞬时
    峰值——方向 = 低估旧流并发（乐观向），窗口盲区 ≤1 epoch。改瞬时
    口径 = 遥测合同侵入（A18'(b) 已否决），此处仅落字披露。

    键两形（冻结接口 ``{link_id: 实测有效速率(B/ns)}`` 的合法键）：

    * **端点形** ``(src_rank, dst_rank)`` 元组 / ``"src->dst"`` 快照字
      符串——与 LinkFlowRegistry 键同空间，参与端点除数合并（上文公
      式的完整形态）；
    * **裸整型 LinkId**（C++ FluidScheduler 链路表下标，C8
      ``_ingest_link_telemetry`` 的原样键）——**端点无关消费**：速率
      查询/transfer_factor 观测/披露计数可用，不参与端点除数合并（注
      册表键是 rank 对，LinkId→有向链路换算需拓扑维序知识，代价模型
      内不可得）。**已知缺口登记（C7 BLOCKED B3）**：换算配方 = C++
      ``MultiDimTopology::connect_dimension`` 的确定性枚举（逐维升序
      遍历 src、connect(src, src+stride[dim], bidirectional) 顺次
      append 正向/反向两条 LinkId）；归 SH 侧换算或后续卡闭合前，整
      型键条目在 ``disclosure()["opaque_link_id_entries"]`` 计数披露
      （可见缺口，非静默不合并）。

    字典中不出现的链路 = 无测量（不抬升，注册表值即有效值）。值域校
    验 fail-closed：速率须为正有限实数；速率超容量（非物理）不拒绝
    但合并下界天然被 max(1, ·) 截住。
    """

    def __init__(
        self,
        registry: LinkFlowRegistry,
        effective_rates: Mapping[Hashable, float],
        *,
        link_capacity_bytes_per_ns: float,
        link_flow_counts: Optional[Mapping[Hashable, float]] = None,
    ) -> None:
        capacity = float(link_capacity_bytes_per_ns)
        if not (capacity > 0) or not math.isfinite(capacity):
            raise JointCostError(
                "link_capacity_bytes_per_ns must be a positive finite "
                f"rate, got {link_capacity_bytes_per_ns!r}")
        # M1：流数键归一（与速率同一键规则），值须 ≥1 有限（时间加权
        # 平均流数物理下界 1——活跃窗口内至少一条流）。
        flow_counts: dict[tuple[int, int], float] = {}
        for key, count in dict(link_flow_counts or {}).items():
            if isinstance(count, bool) or not isinstance(count, (int, float)):
                raise JointCostError(
                    f"link_flow_counts[{key!r}] must be a real number, "
                    f"got {count!r}")
            count_value = float(count)
            if not (count_value >= 1) or not math.isfinite(count_value):
                raise JointCostError(
                    f"link_flow_counts[{key!r}] must be >=1 finite "
                    f"(time-weighted active flow count), got {count!r}")
            flow_counts[self._normalize_link_key(key)] = count_value
        self._registry = registry
        self._capacity = capacity
        self._flow_counts = flow_counts
        self._effective: dict[tuple[int, int], float] = {}
        self._measured: dict[tuple[int, int], float] = {}
        self._opaque_measured: dict[int, float] = {}
        for key, rate in dict(effective_rates).items():
            if isinstance(rate, bool) or not isinstance(rate, (int, float)):
                raise JointCostError(
                    f"link_telemetry_rates[{key!r}] must be a real "
                    f"number, got {rate!r}")
            rate_value = float(rate)
            if not (rate_value > 0) or not math.isfinite(rate_value):
                raise JointCostError(
                    f"link_telemetry_rates[{key!r}] must be a positive "
                    f"finite rate, got {rate_value!r}")
            if isinstance(key, bool):
                raise JointCostError(
                    f"link_telemetry_rates key {key!r} is not a link key "
                    f"(directed (src, dst) tuple, 'src->dst', or bare "
                    f"C++ LinkId int)")
            if isinstance(key, int):
                # 裸整型 LinkId：端点无关消费（见类 docstring 缺口登记）。
                self._opaque_measured[key] = rate_value
                continue
            link = self._normalize_link_key(key)
            self._measured[link] = rate_value
            if link in flow_counts:
                # M1：流数口径除数（时间加权活跃流数 = 物理除数，含
                # collective；无"下游瓶颈"歧义），取代 capacity/合计
                # 速率旧口径（该口径满载退 1、瓶颈误判争用）。
                self._effective[link] = flow_counts[link]
            else:
                self._effective[link] = capacity / rate_value
        # O7③（A19'(g)③，2026-09-23）：混合形态补漏——_effective 键集
        # 扩为 rates ∪ flow_counts 并集（与 JointCostModel.__post_init__
        # 纯仅流数假速率分支同口径）：仅流数链路此前不进 _effective，
        # _union_floor/telemetry_links/max_effective_divisor 漏计（探针
        # 实证：(1,2) 仅流数 7.0 时 divisor_effective=8.0 正确、
        # _union_floor 只给 2.0）。divisor_effective 经 _flow_counts
        # 优先本已正确，此处只补齐下界/披露键集；纯仅流数形态本就完备。
        for link, count in flow_counts.items():
            if link not in self._effective:
                self._effective[link] = count

    # ------------------------------------------------------------ 键 --
    @staticmethod
    def _normalize_link_key(key: Hashable) -> tuple[int, int]:
        """端点形链路键归一：(src, dst) 元组直过；"src->dst" 字符串解
        析；其余 fail-closed（整型 LinkId 在 __init__ 分流，不走此处）。"""
        if isinstance(key, tuple) and len(key) == 2 \
                and all(isinstance(part, int) and not isinstance(
                    part, bool) for part in key):
            return (key[0], key[1])
        if isinstance(key, str):
            parts = key.split("->")
            if len(parts) == 2:
                try:
                    return (int(parts[0]), int(parts[1]))
                except ValueError:
                    pass
        raise JointCostError(
            f"link_telemetry_rates key {key!r} is not a directed link "
            f"key ((src, dst) tuple or 'src->dst')")

    # ---------------------------------------------------------- 查询 --
    def measured_effective_rate(
        self, link: Hashable,
    ) -> Optional[float]:
        """该链路的实测有效速率（bytes/ns）；无测量样本返回 None。

        端点形与整型 LinkId 键均可查询（各自键空间）。"""
        if isinstance(link, bool):
            return None
        if isinstance(link, int):
            return self._opaque_measured.get(link)
        return self._measured.get(self._normalize_link_key(link))

    def effective_divisor(self, link: Hashable) -> Optional[float]:
        """该链路的遥测有效除数；无测量返回
        None（divisor_effective 退化为注册表值）。M1：流数在场取流数
        口径（时间加权活跃流数 = 物理除数），缺席退 capacity/实测速率。"""
        if (not isinstance(link, bool) and not isinstance(link, int)):
            flow_count = self._flow_counts.get(
                self._normalize_link_key(link))
            if flow_count is not None:
                return flow_count
        rate = self.measured_effective_rate(link)
        if rate is None:
            return None
        return self._capacity / rate

    def divisor_effective(
        self, link: Hashable, *, include_self: bool = False,
        self_overlap: int = 0,
    ) -> float:
        """单链路有效除数（N2 叠加口径）= max(注册旧流, 遥测旧流) +
        候选自身占用（include_self 时）。M1：遥测旧流口径优先流数
        （时间加权活跃流数 = 物理除数，含 collective——合计吞吐无法
        反推流数），缺席退 capacity/实测速率。

        N2（2026-09-23 复核审计2）：旧合并式 max(注册含候选, 遥测旧流)
        漏"候选 + 未登记旧流"叠加——遥测不含候选（决策前旧流）、注册
        不含未登记流（collective 等），两未登记旧流加一条候选时实际并
        发 3 模型只算 2。正解：旧流并发两口径（注册 flows / 遥测流数）
        取大者为 base（C7 冻结 max 防漏计保守性保留——哪边更高信哪
        边），候选份额恒叠加。注册腿 = 该链路已登记流数（不含候选；
        候选由 self_overlap 承载，与 LinkFlowRegistry.divisor 单链路
        退化同式）。整型 LinkId 键无注册腿可言（键空间不相交）→ 纯遥
        测腿（≥1，无候选叠加——端点无关消费语境，调用方自担候选份额）。
        """
        if isinstance(link, bool):
            raise JointCostError(f"invalid link key {link!r}")
        if isinstance(link, int):
            telemetry_int = self.effective_divisor(link)
            if telemetry_int is None:
                raise JointCostError(
                    f"no telemetry sample for LinkId {link!r}")
            return float(max(1.0, telemetry_int))
        link_key = self._normalize_link_key(link)
        base_flows = self._registry.registered_flows(
            (link_key[0], link_key[1]))
        telemetry = self.effective_divisor(link)
        base = (base_flows if telemetry is None
                else max(base_flows, telemetry))
        share = base + (self_overlap if include_self else 0)
        return float(max(1, share))

    # ---------------------------------------------------------- 除数 --
    def divisor(
        self, path_ranks: Sequence[int], *, include_self: bool,
        self_overlap: int = 1,
    ) -> float:
        """单路线有效除数（N2 叠加口径）：逐链路 max(注册旧流, 遥测
        旧流) + 候选自身占用（include_self），取路线瓶颈。"""
        if len(path_ranks) < 2:
            return 1.0
        worst = 1.0
        for index in range(len(path_ranks) - 1):
            link = (path_ranks[index], path_ranks[index + 1])
            worst = max(
                worst, self.divisor_effective(
                    link, include_self=include_self,
                    self_overlap=self_overlap))
        return worst

    def divisor_multi(
        self, paths: Sequence[Sequence[int]], *, include_self: bool = True,
    ) -> float:
        """F-B 聚合有效除数（N2 叠加口径）：逐链路 max(注册旧流, 遥测
        旧流) + 候选自身路径经过次数（include_self），取并集瓶颈。

        N2 前 = max(注册并集瓶颈含候选, 遥测下界不含候选)——漏候选与
        未登记旧流的叠加（两未登记流+候选 ⇒ 3 算 2）。"""
        self_counts = _path_union_links(paths)
        if not self_counts:
            return self._union_floor(paths)
        worst = 1.0
        for link, self_share in self_counts.items():
            worst = max(
                worst, self.divisor_effective(
                    link, include_self=include_self,
                    self_overlap=self_share))
        return worst

    def _union_floor(self, paths: Sequence[Sequence[int]]) -> float:
        """路径族并集链路上的遥测有效除数下界（无测量 = 1）。

        O7③（A19'(g)③，2026-09-23）：``_effective`` 键集 = rates ∪
        flow_counts 并集，仅流数链路在下界与披露中同步计入（修前
        混合形态漏计——探针：(1,2) 仅流数 7.0 时本方法只给 1.0）。
        """
        floor = 1.0
        for link in _path_union_links(paths):
            telemetry = self._effective.get(link)
            if telemetry is not None and telemetry > floor:
                floor = telemetry
        return floor

    # ---------------------------------------------------------- 披露 --
    @property
    def collective_coverage(self) -> bool:
        """透传底层注册表的覆盖标记（翻转接线在 SH 侧 C8）。"""
        return self._registry.collective_coverage

    def disclosure(self) -> dict:
        """遥测消费披露（决策日志/manifest 用）。

        ``opaque_link_id_entries`` = 整型 LinkId 键条目数（端点无关消
        费，不参与端点除数合并——可见缺口披露，见类 docstring）。
        """
        return {
            "telemetry_links": len(self._effective),
            "opaque_link_id_entries": len(self._opaque_measured),
            "max_effective_divisor": (
                max(self._effective.values()) if self._effective
                else None),
            "link_capacity_bytes_per_ns": self._capacity,
            "collective_coverage": self._registry.collective_coverage,
            # N3（2026-09-23 复核审计3）：流数覆盖与旧口径退回可辨认——
            # flow_count_links = 有时间加权流数样本的链路数；legacy_
            # rate_divisor_links = 无流数样本而退 capacity/合计吞吐
            # 口径的链路数（旧二进制缺 active_flows 字段的降级可从
            # 披露辨认，不再物理歧义静默）。
            "flow_count_links": len(self._flow_counts),
            "legacy_rate_divisor_links": sum(
                1 for link in self._measured
                if link not in self._flow_counts),
        }


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
    # D1（C1）：prefill 对远端基础历史的实际消费遍数估计。None = 因果
    # 缺省（input_tokens > 0 → 1，否则 0；单遍扫描口径——分块 prefill
    # 的多遍真值属执行侧事实，在线决策不读，由受控测量/后继卡供给）。
    prefill_scan_passes: Optional[int] = None


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
    # credit 形态拆分披露（§4.3.1）：首 credit 块暴露时延 + 其余读流的
    # 纯流送段——合并流（read_passes 全程、时延单计）的全局拆分口径，
    # 日志兼容位；非 remote 动作保持缺省 0（remote_read_ns 恒为全流总
    # 时延口径）。分阶段 prefill/decode 腿量（规格书 §5）不经本冻结
    # schema，走 notes 四键披露（remote_read_prefill_ns / remote_read_
    # decode_ns / suffix_restore_ns / prefill_pipeline_overlap_ns）。
    remote_read_first_credit_ns: int = 0
    remote_read_stream_ns: int = 0


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


def _shard_endpoint_divisors(
    port_registry, shard_paths: Sequence[Sequence[int]],
) -> tuple[int, int]:
    """逐 shard 端点 HBM 端口除数（u_home, u_exec）——C2 供给。

    ``port_registry`` 为 None（离线/单测未注入）时返回 (0, 0)：端点腿
    按独占带宽计。HbmPortFlowRegistry（C2，WL/joint/hbm_port_flow_
    registry.py）注入后按 ``divisor(port_rank)`` 取值：u_home = 源 rank
    端口的 u_port、u_exec = 目标 rank 端口的 u_port（F4 口径：活跃
    decode 消费流 + 在册传输/远读流；**不含候选自身 +1**——消费方
    ``_shard_leg_ns`` 显式加，与 LinkFlowRegistry 的 self_overlap 分离
    口径一致）。
    """
    if port_registry is None:
        return 0, 0
    first_path = shard_paths[0] if shard_paths else ()
    if len(first_path) < 2:
        return 0, 0
    return (
        int(port_registry.divisor(first_path[0])),
        int(port_registry.divisor(first_path[-1])),
    )


def _shard_hop_latency_ns(
    rates: JointHardwareRates, shard_paths: Sequence[Sequence[int]],
) -> int:
    """单 shard 路线的逐跳累计时延（store-and-forward，跳数线性）。"""
    return rates.d2d_latency_ns * max(
        (len(path) - 1 for path in shard_paths if len(path) >= 2), default=0)


def _shard_leg_ns(
    rates: JointHardwareRates,
    noc_effective: float,
    port_registry,
    shard_paths: Sequence[Sequence[int]],
    shard_bytes: int,
    kind: str,
) -> float:
    """单 shard 三腿（read/copy/merge 为两腿、copy/staging 三腿）的最大
    腿时（无 startup/逐跳时延；F1 冻结公式）。"""
    if shard_bytes < 0:
        raise JointCostError("transfer bytes must be >= 0")
    u_home, u_exec = _shard_endpoint_divisors(port_registry, shard_paths)
    legs = (
        # NoC 流送腿：除数 = divisor_multi(候选路径并集)（含候选自身 TP
        # 流与他流；共链自身重叠计费保留，divisor_multi 语义不动）。
        shard_bytes / noc_effective,
        # 源端 HBM 读腿：local_hbm_bytes_per_ns 正式消费（E12）。
        shard_bytes / (rates.local_hbm_bytes_per_ns / (u_home + 1)),
    )
    if kind in ("read", "copy", "staging"):
        # 执行端 HBM 写腿：copy/staging 为目标端留存写（"staging" 为
        # C14/D6 条件项 STAGING_WRITE 复活预留的接口位，未触发前无
        # 消费者）；"read" 为 A4' 补价 rider（PROVENANCE §20.1，C14 G1
        # 裁定 (a)，2026-09-22）——执行侧 remote-read 每 credit 块到达
        # 实计 COMM_WRITE（N-way 严格均分），计价忠实于执行增同式写腿
        # （u_exec 自 hbm_port_registry；read_passes 倍乘已在调用方字节
        # 基数内——F14 无留存暂存 ⇒ 每遍重扫的块真实再到达、再写一次）。
        # kind="merge" 仍无 exec 写腿（C1 冻结：落点空间准备在
        # _eviction_wait_estimate 单列计价）。
        legs = legs + (
            shard_bytes / (rates.local_hbm_bytes_per_ns / (u_exec + 1)),)
    return max(legs)


def _union_noc_divisor(
    flow_registry: "LinkFlowRegistry | TelemetryLinkFlowView",
    *,
    paths_by_rank: Sequence[Sequence[Sequence[int]]],
    bytes_by_rank: Sequence[int],
    include_self: bool,
) -> int:
    """候选全部 TP 并行流链路并集的瓶颈除数（零字节 shard 无流不计）。

    E4 去重（A1 勘误，C2）：同一路径/字节集合的计价腿族共享一次计算
    （调用方经 ``noc_divisor`` 参数复用；breakdown 的
    contention_divisor 同源复用，不再二次调用 divisor_multi）。
    """
    return max(1, flow_registry.divisor_multi(
        [path for paths, shard_bytes in zip(
            paths_by_rank, bytes_by_rank) if shard_bytes > 0
         for path in paths],
        include_self=include_self))


def _transfer_ns_shards(
    rates: JointHardwareRates,
    flow_registry: "LinkFlowRegistry | TelemetryLinkFlowView",
    port_registry,
    *,
    paths_by_rank: Sequence[Sequence[Sequence[int]]],
    bytes_by_rank: Sequence[int],
    include_self: bool,
    kind: str,                        # "read" | "copy" | "merge" | "staging"
    startup_ns: int = 0,
    noc_divisor: Optional[int] = None,
) -> int:
    """逐 shard 三腿 min：wall_s = startup + hop_latency(path_s)
    + max(noc_stream, home_read[, exec_write])；TP 并行取 max over shards。

    C1（F1 冻结）：TP shard 字节向量按 rank 对齐（paths_by_rank[s] =
    rank s 的路径序列、bytes_by_rank[s] = rank s 字节）；除数取候选
    全部 TP 并行流链路并集的瓶颈（divisor_multi，含自身与他流）——
    路径不相交时各 shard 独占链路（wall = 单 shard 时间，非 N×）、
    共链时自身重叠计费保留。聚合字节 ÷ 单链路率的旧串行口径在此
    原语处废除（_transfer_ns 保留为测试旧口径参照，无本模块调用点）。

    noc_divisor（E4 去重，C2）：调用方已按同一路径/字节集合算得的并集
    除数可直接复用，跳过内部重复计算。

    空 shard 契约（F1 小项钉字，2026-09-22）：``paths_by_rank`` 为空
    序列（无 shard）⇒ **返回 0 且 ``startup_ns`` 不计入**——无 shard
    即无传输事务，启动时延属"事务发生"的固定成本，不随不存在的流产生
    （调用方零字节 shard 过滤后自然到达此形态；与
    ``_shard_stream_ns_shards`` 的 ``default=0`` 同语义族）。当前实现
    即此行为（max 聚合循环零次），本 docstring 将其登记为冻结契约。
    """
    if kind not in ("read", "copy", "merge", "staging"):
        raise JointCostError(f"unknown transfer kind {kind!r}")
    if len(paths_by_rank) != len(bytes_by_rank):
        raise JointCostError(
            "paths_by_rank/bytes_by_rank must align by TP rank")
    # 除数并集只计真实发流的 shard（零字节 shard 无流、不争用）。
    if noc_divisor is None:
        noc_divisor = _union_noc_divisor(
            flow_registry, paths_by_rank=paths_by_rank,
            bytes_by_rank=bytes_by_rank, include_self=include_self)
    noc_effective = rates.noc_link_bytes_per_ns / max(1, noc_divisor)
    wall_ns = 0
    for shard_paths, shard_bytes in zip(paths_by_rank, bytes_by_rank):
        if shard_bytes <= 0:
            # M5（2026-09-23 验收审计）：零字节 shard 无流——不计逐跳
            # 传输时延与流腿（与除数并集过滤 ``if shard_bytes > 0`` 同
            # 语义族；不过滤时零字节十跳 shard 仍贡献 hop_latency 抬
            # 高 wall：有数据一跳 10ns 的真实发流被报成 ≈100ns）。
            # startup_ns 保留计入：F1 冻结契约（"shard 在场即事务发
            # 生"，对照锚 test_present_shard_keeps_startup_even_with_
            # zero_bytes）——零字节 shard 的事务固定成本仍在。
            wall_ns = max(wall_ns, int(startup_ns))
            continue
        wall_ns = max(
            wall_ns,
            int(int(startup_ns) + _shard_hop_latency_ns(rates, shard_paths)
                + _shard_leg_ns(
                    rates, noc_effective, port_registry,
                    shard_paths, shard_bytes, kind)))
    return wall_ns


def _shard_stream_ns_shards(
    rates: JointHardwareRates,
    flow_registry: "LinkFlowRegistry | TelemetryLinkFlowView",
    port_registry,
    *,
    paths_by_rank: Sequence[Sequence[Sequence[int]]],
    bytes_by_rank: Sequence[int],
    include_self: bool,
    kind: str,
    noc_divisor: Optional[int] = None,
) -> int:
    """逐 shard 三腿 max 的纯流送段（无 startup/逐跳时延）——credit
    其余读流段（remaining_stream_ns）口径：与 ``_transfer_ns_shards``
    同腿源同除数，仅剥离首块启动时延（§4.3.1 拆分纪律）。noc_divisor
    复用语义同 ``_transfer_ns_shards``（E4 去重，C2）。"""
    if kind not in ("read", "copy", "merge", "staging"):
        raise JointCostError(f"unknown transfer kind {kind!r}")
    if len(paths_by_rank) != len(bytes_by_rank):
        raise JointCostError(
            "paths_by_rank/bytes_by_rank must align by TP rank")
    # 除数并集只计真实发流的 shard（零字节 shard 无流、不争用）。
    if noc_divisor is None:
        noc_divisor = _union_noc_divisor(
            flow_registry, paths_by_rank=paths_by_rank,
            bytes_by_rank=bytes_by_rank, include_self=include_self)
    noc_effective = rates.noc_link_bytes_per_ns / max(1, noc_divisor)
    return max(
        (int(_shard_leg_ns(
            rates, noc_effective, port_registry,
            shard_paths, shard_bytes, kind))
         for shard_paths, shard_bytes in zip(paths_by_rank, bytes_by_rank)),
        default=0)


# Keep this private planner in sync with face_scheduler's C13 planner.  The
# cost model deliberately does not import the runtime scheduler (the latter
# imports this module), but both sides use the same deterministic eight-layer
# target.  A separate helper also makes the byte conservation rule explicit:
# cumulative integer boundaries preserve every rank's original byte total.
_COPY_HANDOFF_CHUNK_LAYERS = 8


def _copy_handoff_layer_ranges(
    prefix_layers: int,
) -> tuple[tuple[int, int], ...]:
    """Return the runtime C13 consumption-order layer chunks."""
    if prefix_layers <= 0:
        return ()
    chunk_count = -(-prefix_layers // _COPY_HANDOFF_CHUNK_LAYERS)
    span = -(-prefix_layers // chunk_count)
    ranges: list[tuple[int, int]] = []
    start = 0
    while start < prefix_layers:
        end = min(start + span, prefix_layers)
        ranges.append((start, end))
        start = end
    return tuple(ranges)


def _copy_handoff_chunk_bytes(
    bytes_by_rank: Sequence[int],
    prefix_layers: int,
    ranges: Sequence[tuple[int, int]],
) -> tuple[tuple[int, ...], ...]:
    """Split each rank's prefix bytes over layer ranges without loss."""
    if prefix_layers <= 0 or not ranges:
        return ()
    if any(int(value) < 0 for value in bytes_by_rank):
        raise JointCostError("copy prefix bytes must be >= 0")
    return tuple(
        tuple(
            int(total) * end // prefix_layers
            - int(total) * start // prefix_layers
            for total in bytes_by_rank)
        for start, end in ranges)


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
    # C1 接口位 / C2 供给：实例 HBM 端口流注册表（u_port 除数，F4）。
    # HbmPortFlowRegistry（WL/joint/hbm_port_flow_registry.py）注入后
    # 端点腿（home_read/exec_write）按 u_home/u_exec+1 计争用；None 时
    # 端点腿独占带宽（离线/单测口径，u_home=u_exec=0）。
    hbm_port_registry: object = None
    # remote-read credit 块大小 K 配置（"auto" | 显式正整数字符串；
    # joint_config 同源）。执行口径 = 逐 credit 交错流唯一机制
    # （2026-09-17 用户裁定：v1 串行加法口径删除，不作为开关可选项
    # 保留），计价为分阶段关键路径（规格书 §5，2026-09-25）：prefill /
    # decode 两腿各按 first_credit + max(remaining_stream, compute) 同
    # 式合成、后缀池恢复并入 prefill 阶段 max——K 与执行侧切片同源
    # （remote_credit_block_size 单一裁决点，两腿共用同一 K），无窗口
    # 矛盾。
    remote_credit_iters: str = "auto"
    # 需求①（2026-09-17《部分层逐出kv管理改造分析方案》）：PARTIAL 基
    # remote-read 适用面消融（JOINT_REMOTE_READ_PARTIAL，on|off 缺省
    # on）。off 只把 PARTIAL 基的 remote-read 移出候选集（copy/recompute
    # 照常参与比较）——与 remote_actions 能力开关同一消融模式，非旧
    # 机制回归档（用户裁定④边界澄清）。
    remote_read_partial: bool = True
    # C7（WP2，冻结交接接口——kwarg 名与 C8 联合冻结）：逐链路遥测实测
    # 有效速率 ``{link_id: 实测有效速率(bytes/ns)}``——SH/C8 解析桥请求
    # ``link_telemetry[]`` 后传入构造（键两形：端点形 (src, dst)/
    # "src->dst" 参与除数合并；裸整型 LinkId 端点无关消费——见
    # TelemetryLinkFlowView）。None/空 = 遥测未开启：NoC 除数走原注册
    # 表模型，行为零漂移（fail-closed 语义在视图构造处）。
    link_telemetry_rates: Optional[Mapping[Hashable, float]] = None
    # M1（2026-09-23 验收审计）：C++ 时间加权活跃流数（含 collective）
    # ——在场时遥测除数取流数口径（物理除数；合计吞吐无法反推流数），
    # 缺席退 link_telemetry_rates 的 capacity/速率旧口径。
    link_telemetry_flow_counts: Optional[Mapping[Hashable, float]] = None
    # N4（2026-09-23 复核审计4）：prefill 整段负载函数（(input_tokens,
    # history_tokens) -> ns）——生产列车按真实 chunk 切分与累计 context
    # 求 roofline（二次形状），线性外推（prefill_ns_per_token × tokens）
    # 丢形状。SH 注入生产同形实现；None 退线性口径（既有测试/离线
    # 兼容）。recompute 的 history 按 ``_resident_here`` 二分（O1，
    # A19'(a)，2026-09-23：与执行侧 R13 双分支同判据——驻留目标传
    # session.history_tokens（span 基=H），异地/REMOTE 副本传 0）。
    prefill_task_load_ns_fn: object = None

    def __post_init__(self) -> None:
        if self.prefill_ns_per_token <= 0 or self.decode_ns_per_token <= 0:
            raise JointCostError("service rates must be positive")
        if self.model_layers <= 0 or self.instance_tp_size <= 0:
            raise JointCostError("layout parameters must be positive")
        # 决策时刻先落盘同时刻汇总样本（P3：读因子前 flush）。
        self.service_factors.flush()
        # C7：遥测叠加视图——NoC 除数消费点统一走 _noc_registry（无遥
        # 测 = 原 flow_registry 引用，零漂移；flow_registry 本体状态不
        # 变，保持决策时点只读快照纪律）。
        if self.link_telemetry_rates:
            self._noc_registry = self.flow_registry.with_effective_rates(
                self.link_telemetry_rates,
                link_capacity_bytes_per_ns=(
                    self.rates.noc_link_bytes_per_ns),
                link_flow_counts=self.link_telemetry_flow_counts)
        elif self.link_telemetry_flow_counts:
            # M1：仅流数在场（速率全零窗被丢弃）也可构造流数口径视图。
            self._noc_registry = self.flow_registry.with_effective_rates(
                {link: 1.0 for link in self.link_telemetry_flow_counts},
                link_capacity_bytes_per_ns=(
                    self.rates.noc_link_bytes_per_ns),
                link_flow_counts=self.link_telemetry_flow_counts)
        else:
            self._noc_registry = self.flow_registry

    def link_telemetry_disclosure(self) -> dict:
        """C7：遥测消费披露（决策日志/manifest 通道；无遥测 = 关闭态
        单键）。"""
        if not isinstance(self._noc_registry, TelemetryLinkFlowView):
            return {"telemetry": False}
        return {"telemetry": True, **self._noc_registry.disclosure()}

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

        N1(a) 解除（2026-09-17《部分层逐出kv管理改造分析方案》需求①）：
        remote-read 适用面从 LOCAL 基扩至 PARTIAL 基——混合形态 = 准入相
        后缀 [p,L) 池恢复物化（热 KV，复用 copy 池腿原语/发射/前递补边/
        空间准备）＋ decode 相前缀 [0,p) credit 读流＋增量驻留执行端；
        LOCAL 基（p==L）语义逐字节不变（回归锚）。REMOTE 基仍拒（无主
        session，裁定③走池恢复/重算就地转正）。适用面消融开关
        ``JOINT_REMOTE_READ_PARTIAL=off`` 时 PARTIAL 基照旧不进候选
        （拒绝理由 "remote-read for partial sessions disabled"）。
        历史：N1 裁定 (a)（2026-09-14）当时的排除是"混合读流原语未建"
        的实现边界而非理论禁令，本案收口（物化/计价锚点沿 R16 层区间
        化纪律；单测 joint/test_joint_mechanisms.py 适用性表）。
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
        # K5（P2-5）处置注记：REMOTE 基 ×copy@home（裁定③就地转正形态）
        # 保持适用——物化侧 merge_back 的 N9 豁免（pool-only 基不 raise，
        # 走 in_place 免单）与本计价（merge_in_place, merge_ns=0）闭合。
        copy_ok = (
            session.location != "none"
            and session.history_tokens > 0
            and not (
                session.resident_instance == instance_index
                and session.location in ("local_hbm", "partial_hbm_remote")))
        results.append((
            copy_ok,
            None if copy_ok else "no history to copy"))
        # remote-read：基础历史驻留前缀在别的 instance（LOCAL 全层
        # 单源读流；PARTIAL 混合形态——后缀准入相池恢复物化＋前缀
        # [0,p) 读流，见 docstring），且 remote 能力开关开启。REMOTE
        # 基（无主）仍拒；PARTIAL 受 JOINT_REMOTE_READ_PARTIAL 消融。
        remote_ok = (
            remote_enabled
            and session.location in ("local_hbm", "partial_hbm_remote")
            and (session.location == "local_hbm"
                 or self.remote_read_partial)
            and session.resident_instance is not None
            and session.resident_instance != instance_index)
        results.append((
            remote_ok,
            None if remote_ok else (
                "remote actions disabled"
                if not remote_enabled
                else "remote-read for partial sessions disabled"
                if session.location == "partial_hbm_remote"
                and not self.remote_read_partial
                else "no resident remote history"
                if session.location != "local_hbm"
                else "base resident at target instance")))
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
        """单候选代价：到 merge_done 边界的完成时间预测（关键路径，
        非机械相加：可重叠段取 max，串行依赖段相加；终点术语 = F11
        冻结的 merge_done——G2 术语 rider，语义零变更）。

        R3'（2026-09-14）：空间准备按动作感知逐 rank 足迹（与 R1' 预约
        足迹同源——stay/recompute@驻留 = final−resident、copy/recompute
        @异地 = 整份、remote-read = input 增量；一律叠加因果 decode 增长
        估计），深缺口按"活跃剩余负载峰值"计价而非一律池写回。
        R15-1/F-B：NoC 除数按候选全部 TP 并行流链路并集取瓶颈；
        R15-3：池路径除数挂池端口仲裁份额。N11：remote-read 计价基数
        补 input + 在线 decode 增长（步数仍为因果估计，因果边界）。
        分阶段关键路径（规格书 §5，2026-09-25）：remote-read 合成改为
        prefill/decode 两阶段——prefill 前缀读流与后缀池恢复同 frontier
        并行分叉、prefill 计算按层段随数据到达推进（无全局 barrier），
        decode 只对 home 前缀新发 credit 读（后缀在 exec HBM 就地复
        用）；旧"后缀恢复全量串在 prefill 计算之前"的合成形态废除，
        阶段量经 notes 披露（详见关键路径合成处注释）。
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
            load, space_needed, notes, instance_index=instance_index,
            covered_wait_ns=target_wait)

        # ---- 3. 历史准备（依动作而异）。----
        history_prep = 0
        remote_read_ns = 0
        # credit 拆分段（仅 ACTION_REMOTE 时被赋值；
        # serial 与非 remote 动作恒 0——breakdown 审计字段同此缺省）。
        first_credit_ns = 0
        remaining_stream_ns = 0
        # 分阶段关键路径腿（规格书 §5，2026-09-25，仅 ACTION_REMOTE 赋
        # 值）：prefill 前缀读流与 decode 前缀 credit 读各自成腿（独立
        # 事务、各含首块启动/逐跳时延），后缀池恢复并行项在关键路径合
        # 成处并入 prefill 阶段 max——后缀恢复不再全量串在计算之前。
        prefill_leg_ns = 0
        prefill_first_credit_ns = 0
        prefill_remaining_stream_ns = 0
        decode_leg_ns = 0
        decode_first_credit_ns = 0
        decode_remaining_stream_ns = 0
        # C13 copy handoff critical-path pieces.  ``history_prep`` remains
        # the complete transfer amount for the frozen audit breakdown; these
        # private values only describe the first ready chunk and the tail
        # stream that can overlap execution.
        copy_first_chunk_ns = 0
        copy_remaining_stream_ns = 0
        copy_pool_suffix_ns = 0
        copy_streaming = False
        # E4 去重（A1，C2）：主计价腿（copy 前缀腿 / remote 读流族）算得
        # 的并集除数——breakdown 的 contention_divisor 复用之；无 NoC
        # 计价腿的动作（stay/recompute）保持 None，函数尾走单次调用。
        # C7：遥测叠加时该值为 float（max(注册, 遥测下界)），计价腿直
        # 用原值；breakdown 披露位函数尾保守取整。
        priced_divisor: Optional[float] = None
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
                # C1/F1：NoC 前缀腿逐 shard 三腿 min（kind="copy" 含
                # exec 留存写腿）。后缀腿守卫：_pool_transfer_ns 对零字节
                # 仍返回端口时延（pool_latency_ns），无守卫会击穿 LOCAL
                # 基与旧值逐位一致（stay 分支 if missing 即先例）。
                # C2/F4：端点 u_port 分解披露（notes 通道）+ E4 并集除数
                # 一次计算（计价腿与 breakdown 共用）。
                copy_paths_by_rank = self._shard_paths(
                    candidate_paths, session.history_bytes_by_tp_rank)
                priced_divisor = _union_noc_divisor(
                    self._noc_registry,
                    paths_by_rank=copy_paths_by_rank,
                    bytes_by_rank=session.history_bytes_by_tp_rank,
                    include_self=True)
                history_prep = _transfer_ns_shards(
                    self.rates, self._noc_registry,
                    self.hbm_port_registry,
                    paths_by_rank=copy_paths_by_rank,
                    bytes_by_rank=session.history_bytes_by_tp_rank,
                    include_self=True, kind="copy",
                    noc_divisor=priced_divisor)
                self._append_u_port_note(notes, copy_paths_by_rank)
                if missing:
                    copy_pool_suffix_ns = _pool_transfer_ns(
                        total_bytes=missing, divisor=pool_divisor,
                        rates=self.rates)
                    history_prep += copy_pool_suffix_ns
                # C13: only the resident prefix handoff is streamed.  The
                # runtime's chunk 0 is on the admission chain; later chunks
                # are side branches whose layer ranges gate compute.  Keep
                # pool suffix restoration on the old serial preparation path.
                prefix_layers = min(
                    max(int(session.resident_prefix_layers), 0),
                    self.model_layers)
                chunk_ranges = _copy_handoff_layer_ranges(prefix_layers)
                chunk_bytes = _copy_handoff_chunk_bytes(
                    session.history_bytes_by_tp_rank,
                    prefix_layers,
                    chunk_ranges)
                if chunk_bytes and any(session.history_bytes_by_tp_rank):
                    copy_streaming = True
                    first_bytes = chunk_bytes[0]
                    first_paths = self._shard_paths(
                        candidate_paths, first_bytes)
                    # Reuse the prefix union divisor already computed for the
                    # candidate; querying the registry again would violate
                    # the single-divisor accounting contract.
                    first_divisor = priced_divisor
                    copy_first_chunk_ns = _transfer_ns_shards(
                        self.rates, self._noc_registry,
                        self.hbm_port_registry,
                        paths_by_rank=first_paths,
                        bytes_by_rank=first_bytes,
                        include_self=True, kind="copy",
                        noc_divisor=first_divisor)
                    if len(chunk_bytes) > 1:
                        tail_bytes = tuple(
                            sum(chunk[rank] for chunk in chunk_bytes[1:])
                            for rank in range(self.instance_tp_size))
                        tail_paths = self._shard_paths(
                            candidate_paths, tail_bytes)
                        # Reuse the primary prefix union divisor.  The
                        # candidate's registered competing flows are already
                        # priced for the full resident prefix; recomputing a
                        # tail-only divisor would double-count the E4
                        # contention query and could make the tail look
                        # artificially faster when a rank has no tail bytes.
                        tail_divisor = priced_divisor
                        copy_remaining_stream_ns = _shard_stream_ns_shards(
                            self.rates, self._noc_registry,
                            self.hbm_port_registry,
                            paths_by_rank=tail_paths,
                            bytes_by_rank=tail_bytes,
                            include_self=True, kind="copy",
                            noc_divisor=tail_divisor)
                    notes.append(f"copy_handoff_chunks={len(chunk_bytes)}")
                    notes.append(
                        f"copy_handoff_first_chunk_ns={copy_first_chunk_ns}")
                    notes.append(
                        "copy_handoff_remaining_stream_ns={}"
                        .format(copy_remaining_stream_ns))
                notes.append("noc_prefix+pool_suffix_restore")
        elif action == ACTION_RECOMPUTE:
            # 无历史搬运；重算缺失区间的时间计入计算段（R13：@驻留仅
            # 缺失后缀折算 token，@异地整份历史；§13.2 复用收益体现为
            # 少做的重算或传输）。
            notes.append("recompute_history_in_compute")
        elif action == ACTION_REMOTE:
            # D1（C1 步骤 2b，设计文档 §1.4）：decode 每步远读基数 =
            # 仍位于远端的基础历史（history_bytes_by_tp_rank——PARTIAL
            # 基即驻留前缀 [0,p)）；执行端生成的 input/decode 增量本地
            # 读取、不计作远读，prefill 读流对基础历史按实际消费遍数计
            # （read_passes = prefill_scan_passes + decode 步数）。旧口径
            # "终态全部上下文 × 每步"（N11 的 input+growth 并入）废除。
            # 需求①混合形态（2026-09-17）：PARTIAL 基后缀 [p,L) 准入相
            # 池恢复物化仍计入 history_prep（与 ACTION_COPY 池腿同源同
            # 口径）；LOCAL 基（p==L）missing==0 无池腿（回归锚）。
            missing = sum(session.missing_bytes_by_tp_rank)
            if session.location == "partial_hbm_remote" and missing:
                # 后缀池腿守卫：_pool_transfer_ns 零字节时延陷阱
                # （copy 分支同因）。
                history_prep = _pool_transfer_ns(
                    total_bytes=missing, divisor=pool_divisor,
                    rates=self.rates)
                notes.append("pool_suffix_restore_hybrid")
            if session.location == "partial_hbm_remote":
                read_prefix_layers = min(
                    max(session.resident_prefix_layers, 0),
                    self.model_layers)
                notes.append(
                    f"remote_read_prefix_layers={read_prefix_layers}")
            # 读流逐 shard 化（kind="read" 三腿：A4' 补价 rider 后含
            # 执行端写腿——镜像执行侧 COMM_WRITE；F14 容量半边不动——
            # credit 在途字节仍不占执行端 HBM 容量、暂存归还仍无操作）。
            # C2/F4：端点 u_port 分解披露（notes 通道）+ E4 并集除数
            # 一次计算（全流/首 credit/其余流送三处腿共享——同路径集、
            # 同非零过滤，乘数为正标量）。
            read_base_by_rank = session.history_bytes_by_tp_rank
            shard_paths = self._shard_paths(
                candidate_paths, read_base_by_rank)
            self._append_u_port_note(notes, shard_paths)
            prefill_scans, decode_steps = self._remote_read_passes(request)
            read_passes = prefill_scans + decode_steps
            if read_passes:
                priced_divisor = _union_noc_divisor(
                    self._noc_registry,
                    paths_by_rank=shard_paths,
                    bytes_by_rank=read_base_by_rank,
                    include_self=True)
            remote_read_ns = _transfer_ns_shards(
                self.rates, self._noc_registry, self.hbm_port_registry,
                paths_by_rank=shard_paths,
                bytes_by_rank=tuple(
                    shard_bytes * read_passes
                    for shard_bytes in read_base_by_rank),
                include_self=True, kind="read",
                noc_divisor=priced_divisor,
            ) if read_passes else 0
            notes.append(f"prefill_scan_passes={prefill_scans}")
            notes.append(f"remote_read_passes={read_passes}")
            # credit 拆分（§4.3.1 唯一执行口径；C1 逐 shard 化）：读流
            # 按 shard 字节分布切 credit 块——首块暴露时延与其余读流纯
            # 流送段均逐 shard 取 max（同腿源同除数）；K 与执行侧切片
            # 同源（remote_credit_block_size 单一裁决点，逻辑不动）。
            # 不变量：K >= read_passes（单 credit）时 first_credit_ns ==
            # remote_read_ns、remaining_stream_ns == 0 ⇒ 合成退化为旧
            # 加法形态的数值。披露字段口径不变：remote_read_ns 恒为
            # 全流总时延口径。
            if read_passes:
                credit_k = remote_credit_block_size(
                    self.remote_credit_iters, read_passes)
                credit_steps = min(credit_k, read_passes)
                first_credit_ns = _transfer_ns_shards(
                    self.rates, self._noc_registry,
                    self.hbm_port_registry,
                    paths_by_rank=shard_paths,
                    bytes_by_rank=tuple(
                        shard_bytes * credit_steps
                        for shard_bytes in read_base_by_rank),
                    include_self=True, kind="read",
                    noc_divisor=priced_divisor)
                remaining_stream_ns = _shard_stream_ns_shards(
                    self.rates, self._noc_registry,
                    self.hbm_port_registry,
                    paths_by_rank=shard_paths,
                    bytes_by_rank=tuple(
                        shard_bytes * (read_passes - credit_steps)
                        for shard_bytes in read_base_by_rank),
                    include_self=True, kind="read",
                    noc_divisor=priced_divisor)
                notes.append(f"remote_credit_k={credit_k}")
                # 分阶段关键路径腿（规格书 §5，2026-09-25）：prefill 腿
                # = 前缀 [0,p) 读流（prefill_scans 遍），decode 腿 =
                # decode 对 home 前缀新发的 credit 读（decode_steps 遍；
                # 后缀 [p,L) 已在 exec HBM 就地复用、不再从池恢复）。
                # 两腿为独立事务、各含首块启动/逐跳时延，其余为纯流送段
                # （与全局拆分同纪律）；切块沿用同一 K
                # （remote_credit_block_size 单一裁决点），腿字节基 =
                # 同一 read_base 逐 shard 向量 ⇒ 并集除数同源复用（E4
                # 单次计算契约不变）。零遍腿恒 0（零字节 shard 契约）。
                prefill_credit_steps = min(credit_k, prefill_scans)
                decode_credit_steps = min(credit_k, decode_steps)
                prefill_leg_ns = _transfer_ns_shards(
                    self.rates, self._noc_registry,
                    self.hbm_port_registry,
                    paths_by_rank=shard_paths,
                    bytes_by_rank=tuple(
                        shard_bytes * prefill_scans
                        for shard_bytes in read_base_by_rank),
                    include_self=True, kind="read",
                    noc_divisor=priced_divisor)
                prefill_first_credit_ns = _transfer_ns_shards(
                    self.rates, self._noc_registry,
                    self.hbm_port_registry,
                    paths_by_rank=shard_paths,
                    bytes_by_rank=tuple(
                        shard_bytes * prefill_credit_steps
                        for shard_bytes in read_base_by_rank),
                    include_self=True, kind="read",
                    noc_divisor=priced_divisor)
                prefill_remaining_stream_ns = _shard_stream_ns_shards(
                    self.rates, self._noc_registry,
                    self.hbm_port_registry,
                    paths_by_rank=shard_paths,
                    bytes_by_rank=tuple(
                        shard_bytes * (prefill_scans - prefill_credit_steps)
                        for shard_bytes in read_base_by_rank),
                    include_self=True, kind="read",
                    noc_divisor=priced_divisor)
                decode_leg_ns = _transfer_ns_shards(
                    self.rates, self._noc_registry,
                    self.hbm_port_registry,
                    paths_by_rank=shard_paths,
                    bytes_by_rank=tuple(
                        shard_bytes * decode_steps
                        for shard_bytes in read_base_by_rank),
                    include_self=True, kind="read",
                    noc_divisor=priced_divisor)
                decode_first_credit_ns = _transfer_ns_shards(
                    self.rates, self._noc_registry,
                    self.hbm_port_registry,
                    paths_by_rank=shard_paths,
                    bytes_by_rank=tuple(
                        shard_bytes * decode_credit_steps
                        for shard_bytes in read_base_by_rank),
                    include_self=True, kind="read",
                    noc_divisor=priced_divisor)
                decode_remaining_stream_ns = _shard_stream_ns_shards(
                    self.rates, self._noc_registry,
                    self.hbm_port_registry,
                    paths_by_rank=shard_paths,
                    bytes_by_rank=tuple(
                        shard_bytes * (decode_steps - decode_credit_steps)
                        for shard_bytes in read_base_by_rank),
                    include_self=True, kind="read",
                    noc_divisor=priced_divisor)
        else:  # pragma: no cover - ACTION_ORDER 封闭
            raise JointCostError(f"unknown action {action!r}")

        # ---- 4. 计算（roofline 基础 × 在线因子；recompute 追加重算
        # 缺失区间的工作量——R13：@驻留目标按池后缀折算 token、@异地
        # 按整份历史；不含排队——排队在 target_wait/eviction 段）。
        prefill_tokens = request.input_tokens
        if action == ACTION_RECOMPUTE:
            prefill_tokens += self._recompute_missing_tokens(
                session, instance_index)
        if self.prefill_task_load_ns_fn is not None:
            # N4：生产同形整段求值（chunk 切分 + 累计 context roofline）。
            # O1（A19'(a)，2026-09-23）：recompute span 基按 resident_here
            # 二分——@驻留目标基 = history_tokens（R13 双分支的驻留腿：
            # @home 重算仍对驻留前缀 KV 做 attention，roofline attention
            # 项 ∝ chunk×context，基 0 丢掉整项、低估 1.12–3.7×；与执
            # 行侧 sh30 ``_recompute_missing_tokens`` 的 resident_here
            # 判据同字段同值，prefix=L 时与 stay 逐位同值）；异地 /
            # REMOTE 副本基 = 0（跨实例物化从零计工作上下文，R13 异地腿
            # 不动）。其余动作真实历史基数。
            if action == ACTION_RECOMPUTE:
                history_base = (
                    session.history_tokens
                    if self._resident_here(session, instance_index)
                    else 0)
            else:
                history_base = session.history_tokens
            prefill_base_ns = int(self.prefill_task_load_ns_fn(
                prefill_tokens, history_base))
        else:
            prefill_base_ns = (
                prefill_tokens * self.prefill_ns_per_token)
        # 分阶段关键路径的计算分量（规格书 §5，2026-09-25）：prefill /
        # decode 两段各自进入所属阶段的重叠 max；compute_ns 审计字段保
        # 持整段口径（C5 冻结字段，数值表达式不动）。
        prefill_compute_ns = int(
            prefill_base_ns * self.service_factors.prefill_factor)
        decode_compute_ns = int(
            request.estimated_decode_tokens
            * self.decode_ns_per_token
            * self.service_factors.decode_factor)
        compute_ns = int(
            prefill_base_ns
            * self.service_factors.prefill_factor
            + request.estimated_decode_tokens
            * self.decode_ns_per_token
            * self.service_factors.decode_factor)

        # ---- 5. compute_done 后的合并（merge v2 少并多，2026-09-17
        # 用户裁定《部分层逐出kv管理改造分析方案》需求②：两侧保留量比
        # 大小、小的整份搬给大的、合并零池写、结果恒全层 LOCAL@胜者、
        # home 迁移胜者；旧机制（增量拆分归并/R4 自降级/k=0 池归并/
        # REMOTE 写池归并）直接删除、单实现——裁定④，无开关可选项）。
        # 计价与物化同判据同公式：merge_ns = min(前向, 反向)；物化按实际
        # 增量 I（已观测 decode 长度），计价按因果估计 I_pred——边界
        # 差异与 K4 同源披露。REMOTE 基 = 无主就地保留（裁定③，
        # merge_ns=0，不写池）；copy/recompute 执行端恒持并集 → 反向
        # 零字节翻转（merge_ns=0、无空间准备）。LOCAL 基 remote-read
        # 前向腿 = noc(增量) + home 空间准备等待，与旧公式逐位一致
        # （p==L 退化锚）。
        merge_ns = 0
        if session.home_instance is not None and (
                session.home_instance != instance_index
                or session.location == "remote_memory"):
            if session.location == "remote_memory":
                # 无主 session：合并就地保留（新 KV 不写池、副本不
                # 释放），零传输零池写。
                notes.append("merge_in_place")
            else:
                increments_by_rank = tuple(
                    input_bytes + growth
                    for input_bytes, growth in zip(
                        request.input_kv_bytes_by_tp_rank,
                        decode_growth_ranks))
                if action == ACTION_REMOTE:
                    # 执行侧保留量 = 增量（LOCAL 基）或 池恢复后缀＋
                    # 增量（PARTIAL 混合形态）；home 侧保留量 = 驻留
                    # 前缀真值（history_bytes_by_tp_rank）。
                    if session.location == "partial_hbm_remote":
                        exec_retained_by_rank = tuple(
                            missing + increment
                            for missing, increment in zip(
                                session.missing_bytes_by_tp_rank,
                                increments_by_rank))
                    else:
                        exec_retained_by_rank = increments_by_rank
                    home_retained_by_rank = tuple(
                        session.history_bytes_by_tp_rank)
                    # C1/F1：merge 双腿逐 shard 三腿 min（kind="merge"：
                    # 无 exec 写腿——落点空间准备在 _eviction_wait_estimate
                    # 单列计价）。K5（P2-1，2026-09-23 外部审计）：反向腿
                    # 改用自身方向（home→exec）的路线集合计价——原实现
                    # 两腿共用前向路线（"方向差异由流登记表的调用方覆盖"
                    # 无对应实现），反向恰是 charged 腿时自身重叠份额/
                    # 已登记他流/端口除数全按错误方向计。
                    forward_paths_by_rank = self._shard_paths(
                        self._route_paths_pair(
                            instance_index, session.home_instance),
                        exec_retained_by_rank)
                    reverse_paths_by_rank = self._shard_paths(
                        self._route_paths_pair(
                            session.home_instance, instance_index),
                        home_retained_by_rank)
                    forward_ns = _transfer_ns_shards(
                        self.rates, self._noc_registry,
                        self.hbm_port_registry,
                        paths_by_rank=forward_paths_by_rank,
                        bytes_by_rank=exec_retained_by_rank,
                        include_self=True, kind="merge")
                    reverse_ns = _transfer_ns_shards(
                        self.rates, self._noc_registry,
                        self.hbm_port_registry,
                        paths_by_rank=reverse_paths_by_rank,
                        bytes_by_rank=home_retained_by_rank,
                        include_self=True, kind="merge")
                    # 胜者侧空间准备等待（与物化 _ensure_capacity 同源：
                    # 前向胜者=home 备 exec 侧保留量；反向胜者=exec 备
                    # home 侧保留量）。
                    home_load = self.loads.get(session.home_instance)
                    if home_load is not None:
                        forward_ns += self._eviction_wait_estimate(
                            home_load, exec_retained_by_rank, notes,
                            instance_index=session.home_instance,
                            prefix="home_merge")
                    exec_load = self.loads.get(instance_index)
                    if exec_load is not None and any(home_retained_by_rank):
                        reverse_ns += self._eviction_wait_estimate(
                            exec_load, home_retained_by_rank, notes,
                            instance_index=instance_index,
                            prefix="exec_merge")
                    # K5（P1-④）：方向判据镜像物化侧（face merge_back：
                    # Σ working ≤ Σ home → forward，F15 平局含等号取
                    # forward）——原 min(前向, 反向) 是另一套判据，
                    # "保留量接近 + home 侧空间紧张"场景系统性低估
                    # remote-read（argmin 偏乐观）。物化的 KVCapacityError
                    # 容量兜底翻转在决策时点不可计价，note 披露。
                    selected_forward = (
                        sum(exec_retained_by_rank)
                        <= sum(home_retained_by_rank))
                    merge_ns = int(
                        forward_ns if selected_forward else reverse_ns)
                    notes.append(
                        "merge_v2_direction=forward" if selected_forward
                        else "merge_v2_direction=reverse")
                    notes.append(
                        "merge_v2_direction_rule=byte_le_tie_forward")
                    notes.append(
                        f"merge_v2_forward_ns={int(forward_ns)}")
                    notes.append(f"merge_v2_reverse_ns={int(reverse_ns)}")
                    notes.append(
                        f"merge_to_home={session.home_instance}")
                else:
                    # copy/recompute：执行端恒持并集（copy 驻留前缀+池
                    # 恢复后缀 / recompute 重算复份 ＋增量）⊇ home 侧
                    # → 反向零字节翻转：零传输、零池写、无空间准备。
                    merge_ns = 0
                    notes.append("merge_v2_zero_byte_flip")

        # ---- 关键路径合成：目标等待 -> 空间准备（驱逐等待，准入前串
        # 行）-> 阶段关键路径 -> 合并。remote-read 分阶段关键路径（规格
        # 书 §5，2026-09-25）：prefill 前缀读流（home→exec）与后缀池恢
        # 复（pool→exec HBM）自同一准入 frontier 并行分叉，prefill 计
        # 算按层段等待对应数据——[0,p) 段等前缀首 credit 到达即可起步、
        # [p,L) 段等恢复到达，无"全量后缀恢复完成才开始 prefill"的全局
        # barrier：
        #   prefill_stage = prefix_first_credit
        #                   + max(prefix_remaining_stream, suffix_restore,
        #                         prefill_compute)
        #   decode_stage  = decode_first_credit
        #                   + max(decode_remaining_stream, decode_compute)
        #   total = target_wait + eviction_wait + prefill_stage
        #           + decode_stage + merge
        # 旧合成 max(history_prep, eviction_wait) + first_credit
        # + max(remaining_stream, compute)（后缀恢复全量串在 prefill 计
        # 算之前的形态）废除；可重叠段取 max、串行依赖段相加的纪律不变。
        # 阶段量经 notes 披露（C5 冻结 schema 不动）：remote_read_prefill_
        # ns / remote_read_decode_ns / suffix_restore_ns /
        # prefill_pipeline_overlap_ns（最后者 = prefill 阶段三并行项之和
        # 被 max 吸收的隐藏量）。breakdown 旧字段口径保持：history_prep_
        # ns 仍为后缀池恢复全量审计值、remote_read_first_credit_ns /
        # remote_read_stream_ns 仍为合并流（read_passes 全程、时延单
        # 计）的全局 credit 拆分——仅作日志兼容披露，不再进关键路径。
        if action == ACTION_REMOTE:
            suffix_restore_ns = history_prep
            prefill_stage_ns = prefill_first_credit_ns + max(
                prefill_remaining_stream_ns, suffix_restore_ns,
                prefill_compute_ns)
            decode_stage_ns = decode_first_credit_ns + max(
                decode_remaining_stream_ns, decode_compute_ns)
            prefill_pipeline_overlap_ns = (
                prefill_remaining_stream_ns + suffix_restore_ns
                + prefill_compute_ns
                - max(prefill_remaining_stream_ns, suffix_restore_ns,
                      prefill_compute_ns))
            notes.append(f"remote_read_prefill_ns={prefill_leg_ns}")
            notes.append(f"remote_read_decode_ns={decode_leg_ns}")
            notes.append(f"suffix_restore_ns={suffix_restore_ns}")
            notes.append(
                f"prefill_pipeline_overlap_ns={prefill_pipeline_overlap_ns}")
            cost = (target_wait + eviction_wait
                    + prefill_stage_ns + decode_stage_ns + merge_ns)
        elif copy_streaming:
            # Chunk 0 plus any pool suffix is the first ready point.  Tail
            # chunks overlap the compute region exactly as the runtime body
            # gates their layer segments; ``history_prep_ns`` above remains
            # the complete transfer audit value.
            prep = max(
                copy_first_chunk_ns + copy_pool_suffix_ns,
                eviction_wait)
            effective_compute = max(
                copy_remaining_stream_ns, compute_ns)
            cost = target_wait + prep + effective_compute + merge_ns
        else:
            effective_compute = (
                first_credit_ns + max(remaining_stream_ns, compute_ns))
            prep = max(history_prep, eviction_wait)
            cost = target_wait + prep + effective_compute + merge_ns

        # E4 去重（A1 勘误，C2）：contention_divisor 复用主计价腿已算得
        # 的并集除数（原此处对 candidate_paths 的第二次 divisor_multi
        # 调用删除）；stay/recompute 无 NoC 计价腿可复用时维持单次调用
        # （非重复——该动作族原本也只有这一次）。
        if priced_divisor is None:
            priced_divisor = max(1, self._noc_registry.divisor_multi(
                candidate_paths, include_self=True))
        divisor = priced_divisor
        # C7：遥测合并除数可为非整数 float；breakdown 披露位保持 int
        # （C5 冻结 schema），向上取整（保守侧）——计价腿全程用原值，
        # 本取整只影响披露字段。
        if not isinstance(divisor, int):
            divisor = int(math.ceil(divisor))
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
                # credit 拆分披露（日志兼容口径）：两字段仍为合并流
                # （read_passes 全程、时延单计）的全局 credit 拆分，
                # serial / 非 remote 动作下恒 0；分阶段 prefill/decode
                # 腿量不走本冻结 schema，经 notes 四键披露（见关键路径
                # 合成处注释）。
                remote_read_first_credit_ns=first_credit_ns,
                remote_read_stream_ns=remaining_stream_ns,
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

    def _shard_paths(
        self,
        candidate_paths: Sequence[Sequence[int]],
        bytes_by_rank: Sequence[int],
    ) -> tuple[tuple[Sequence[int], ...], ...]:
        """逐 rank 路径对齐（``paths_by_rank`` 口径，C1）。

        route_paths_fn 已注入时逐 rank 一致（每 rank 单路径）；未注入
        （单测/离线代表路径口径，_route_paths_pair 返回单元素）时代表
        路径广播到全部 rank——共链自身重叠按 rank 数计，与旧聚合口径
        在字节均衡、空链路时同 wall 值（K8 订正：有背景流时新旧比 =
        (f+n)/(n·(f+1)) ≤ 1，广播口径相对旧聚合口径**偏乐观**——原
        "保守方向"标注方向写反；route_paths_fn 生产恒注入，此口径仅
        测试/离线代表路径可达）。
        """
        size = len(bytes_by_rank)
        if not candidate_paths:
            return ((),) * size
        if len(candidate_paths) == size:
            return tuple((path,) for path in candidate_paths)
        representative = (candidate_paths[0],)
        return tuple(representative for _ in range(size))

    def _append_u_port_note(self, notes: list, shard_paths) -> None:
        """C2/F4：u_port 分解披露 → 决策日志 notes 通道（C5 冻结 schema
        不动——正式字段位 port_snapshot 的接通归 C11 步骤 6）。

        取首 shard 路径的端点端口（rank），分解 = 活跃 decode 消费流数
        ＋ 在册传输/远读流数（HbmPortFlowRegistry.decomposition）。仅
        主计价腿（copy 前缀腿 / remote 读流族）披露；未注入注册表
        （离线/单测）时不披露。
        """
        registry = self.hbm_port_registry
        if registry is None:
            return
        rank_paths = shard_paths[0] if shard_paths else ()
        first_path = rank_paths[0] if rank_paths else ()
        if len(first_path) < 2:
            return
        home_rank, exec_rank = first_path[0], first_path[-1]
        home_active, home_flows = registry.decomposition(home_rank)
        exec_active, exec_flows = registry.decomposition(exec_rank)
        notes.append(
            "u_port_home=r{}:{}act+{}fl".format(
                home_rank, home_active, home_flows))
        notes.append(
            "u_port_exec=r{}:{}act+{}fl".format(
                exec_rank, exec_active, exec_flows))

    def _remote_read_passes(self, request: RequestView) -> tuple[int, int]:
        """D1 读放大因果口径：远读遍数 = prefill 消费遍数 + decode 步数。

        prefill 遍数来源 = ``RequestView.prefill_scan_passes``（执行侧
        /受控测量供给）；缺省因果估计 = input_tokens > 0 ? 1 : 0（单遍
        扫描）。decode 步数 = 因果时域估计（CausalHorizonEstimator 同
        源）。两分量均非 oracle：不读真实输出长度。
        """
        if request.prefill_scan_passes is not None:
            prefill_scans = max(0, int(request.prefill_scan_passes))
        else:
            prefill_scans = 1 if request.input_tokens > 0 else 0
        return prefill_scans, max(0, request.estimated_decode_tokens)

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

    @staticmethod
    def _resident_here(
        session: SessionKVView, instance_index: int,
    ) -> bool:
        """R13/O1：基础历史驻留目标判据——``resident_instance == 目标``
        ∧ ``location ∈ {local_hbm, partial_hbm_remote}``。与执行侧
        sh30_online_scheduler.py recompute 分支（R13 双分支）同字段同
        值；A7' 起 ``_recompute_missing_tokens`` 与本判据同源，O1
        （A19'(a)，2026-09-23）起 roofline span 基消费点并入同一判据
        （修前 span 基把 R13 异地腿的"副本 0 基"误泛化到驻留腿）。
        """
        return (
            session.resident_instance == instance_index
            and session.location in ("local_hbm", "partial_hbm_remote"))

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
        remote-read：LOCAL 基仅 input 增量；PARTIAL 混合形态 = 池恢复
        后缀 [p,L) 物化足迹（热 KV，需求①）＋ input 增量。
        """
        size = self.instance_tp_size
        # Keep the physical TP partition supplied by the KV ledger.  Rebuilding
        # a per-rank vector from its aggregate total silently erases skewed
        # shards (and can move a capacity failure from one rank to another).
        # Every caller in the production path supplies one entry per relative
        # TP rank; reject malformed views instead of letting ``zip`` truncate
        # them and producing a plausible but incomplete footprint.
        vectors = (
            ("history_bytes_by_tp_rank", session.history_bytes_by_tp_rank),
            ("missing_bytes_by_tp_rank", session.missing_bytes_by_tp_rank),
            ("input_kv_bytes_by_tp_rank", request.input_kv_bytes_by_tp_rank),
        )
        for name, values in vectors:
            if len(values) != size:
                raise JointCostError(
                    f"{name} length {len(values)} does not match "
                    f"instance TP size {size}")
            if any(int(value) < 0 for value in values):
                raise JointCostError(f"{name} contains a negative byte count")
        final_ranks = tuple(
            int(history) + int(missing) + int(input_bytes)
            for history, missing, input_bytes in zip(
                session.history_bytes_by_tp_rank,
                session.missing_bytes_by_tp_rank,
                request.input_kv_bytes_by_tp_rank,
            )
        )
        resident_here = self._resident_here(session, instance_index)
        if action == ACTION_REMOTE:
            if session.location == "partial_hbm_remote":
                return tuple(
                    missing + increment
                    for missing, increment in zip(
                        session.missing_bytes_by_tp_rank,
                        request.input_kv_bytes_by_tp_rank))
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

        A7'（2026-09-22，§4.3 补遗）：``resident_here`` 判据自
        ``location == "local_hbm"`` 单态修正为 LOCAL/PARTIAL 双态（对齐
        执行侧 sh30_online_scheduler.py recompute 分支与
        ``_space_footprint_by_tp_rank`` 的既有双态口径）——修正前
        PARTIAL 驻留场景下本函数 partial_hbm_remote 的缺失后缀分支为
        死分支，恒按整份 history 计费，recompute 工作量被系统性高估约
        2 倍、动作选择偏离（计价忠实于执行：SH 侧重算工作量用含
        partial 的正确公式）。
        驻留量折算 = 前缀比例，缺失侧取整：missing =
        history − floor(H×prefix/L)（等价实现 = ceil(H×(L−prefix)/L)，
        ceil 在缺失侧、与执行侧逐字节一致）。
        """
        layers = self.model_layers
        history = session.history_tokens
        if history <= 0:
            return 0
        resident_here = self._resident_here(session, instance_index)
        if resident_here and session.resident_prefix_layers >= layers:
            return 0
        if resident_here:
            # A14'（H8，2026-09-22 第三轮复审）：驻留折算对 resident_
            # here ∧ prefix<L 恒 ceil——去掉 partial_hbm_remote 额外门
            # （LOCAL 基且 prefix<L 形态被"转 LOCAL 恒置 prefix=layers"
            # 不变量排除、生产不可达；原该形态落穿 return history 整份
            # 计费，与"和执行侧逐字节一致"的宣称不符——对齐后不可达
            # 形态也走与执行侧相同的 ceil 公式）。
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
        covered_wait_ns: int = 0,
    ) -> int:
        """空间缺口 -> 时间换算（不把无单位容量风险加到延迟上）。

        R3'（2026-09-14）：逐 rank 精确缺口（调用方给逐 rank 足迹）。
        缺口可由逐出 inactive 解决（needed ≤ reclaimable）→ 池写回
        搬运时间（除数 = 池端口仲裁份额）；深缺口（超出 reclaimable，
        只有活跃完成才能释放）→ ``max(写回估计, 活跃任务剩余负载)``，
        并注记 ``deep_gap_unresolved``（docstring 承诺的观测落盘）——
        §13.1"驱逐压力换算成时间"逐字对齐，因果可观测、无 oracle。

        K5（2026-09-23 外部审计）两处口径修正：
        - 缺省空 ``reclaimable_bytes_by_tp_rank``（视图缺省构造/测试
          夹具不供给）按"零可回收"处理（保守侧）——原 ``and
          reclaimable`` 守卫把"无信息"当"全部可回收"，静默关闭深缺口
          升级（生产 SH 恒显式供给，缺省臂不可达）。
        - ``covered_wait_ns``：已在关键路径前段支付的等待（主调用侧 =
          target_wait——活跃任务排空视界 ⊆ 台账总量）。深缺口分量取
          ``max(0, active_remaining − covered)``，消除与 target_wait
          的 running+active 双计（原加法合成重复计、高估）；merge 两
          腿（他实例空间准备）传 0，语义不变。
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
            if gap > 0:
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
            # rank 同账）为因果可见的等待下界；covered_wait_ns 已含的
            # 部分不得双计（K5）。注记语义 = 深缺口条件成立（主调用侧
            # 分量被 target_wait 覆盖时等待不抬升、注记仍在——"深缺口
            # 存在"与"额外等待"两事实分列）。
            active_remaining = max(
                0,
                load.running_task_load_ns
                + load.active_decode_task_load_ns
                - max(0, int(covered_wait_ns)))
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
