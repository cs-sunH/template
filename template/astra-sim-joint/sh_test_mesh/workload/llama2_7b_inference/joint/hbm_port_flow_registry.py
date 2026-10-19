"""hbm_port_flow_registry.py -- C2（WP1b，2026-09-22）：实例 HBM 端口流
注册表（F4 冻结口径的 u_port 除数执行器）。

u_port(port) = 该端口所属实例的**活跃 decode KV 消费流数**（因果负载
视图派生——SH 侧经 ``attach_active_decode_provider`` 注入读取器，闲置
为 0，不假设恒 1）＋ **在册指向该端口的传输流数**（noc_migrate
发射时登记、完成事件注销）。API 与 LinkFlowRegistry
（joint_cost_model.py）同构：``register(port_id, owner)`` /
``unregister(port_id, owner)`` / ``divisor(port_id)`` / ``snapshot()``；
owner 沿用 SH 既有 owner 字符串约定（``rid``、``rid#decode``、
``rid#decode#{j}``、``rid#merge``）。

除数口径：``divisor(port_id)`` 返回在册他流数（活跃 decode ＋ 在册
传输），**不含候选自身 +1**——消费方（JCM
``_shard_endpoint_divisors`` → ``_shard_leg_ns``）在计价时显式加 1
（``B_HBM/(u+1)``），与 LinkFlowRegistry.divisor 把 self_overlap 分离
给调用方的口径一致。

登记覆盖边界（与 SH ``_register_transfer_flows`` 的腿型对齐）：
noc_migrate 逐 shard 登记**两个端点端口**——源端口（home 读腿）恒有；
目标端口（exec 写腿）对 copy 留存写 / merge 落点写恒有；
D04（2026-09-26，v2.0 C11）起 pool 路径端点同样入表——remote_load
登记 noc_path 末端**目标 HBM 写端点**、remote_store 登记 noc_path
首端**源 HBM 读端点**（edge 侧是池端口、非 rank HBM；F3 旧豁免废
除，候选池腿式 (7) 端点除数自此看见在途池流）；中间跳 rank 不触碰
端点 HBM、不登记。池端口本身（edge rank 的池事务占用）仍由 SH
_PoolPortRegistry 独占（计价侧经 ``pool_edge_flows_fn`` 注入 JCM）。

保真边界披露（F4，PROVENANCE 落字）：u_port 不含权重/激活读——短上下
文权重受限区间端口模型系统性乐观；KV-bound 区间近似正确。端口无遥测，注册表即执行器（遥测校正登记为可选后续）。

D06（2026-09-27，v2.0 §9.1 式(6)）：候选阶段的临时 HBM 占用走
``CandidateStageHbmShare`` 只读叠加视图（本文件尾部）——在册工作仍
由本表独家描述，候选占用另加、不写本表（不污染真实登记）。
"""

from __future__ import annotations

from typing import Callable, Mapping, Optional


class HbmPortFlowError(ValueError):
    """fail-closed：双释放/未知归属等端口登记表破损。"""


class HbmPortFlowRegistry:
    """单 rank HBM 端口的在途流登记表（键 = rank 端口；纯内存状态）。

    生命周期与 LinkFlowRegistry 同族：``register`` 在流发射时登记（携带
    owner 归属键），``unregister``/``release_owner`` 在完成事件（drain /
    decode 完成 / 逐切片核销 / merge watch）注销——登记/注销全部由已
    观测完成事件驱动，决策时刻快照因果可见。活跃 decode 分量不经登记
    通道（其开始/结束边界与流发射不同源），由因果负载视图 provider 在
    读取时派生，杜绝漏更新。
    """

    def __init__(self) -> None:
        self._counts: dict[int, int] = {}
        self._owners: dict[str, list[int]] = {}
        self._active_decode_provider: Optional[Callable[[int], int]] = None
        self.has_registrations = False

    # ------------------------------------------------------------ 供给 --
    def attach_active_decode_provider(
        self, provider: Callable[[int], int],
    ) -> None:
        """活跃 decode KV 消费流数的因果来源（port_id -> 流数）。

        provider 必须是只读快照读取（SH 侧 = len(state.active_decode)
        经 rank→instance 映射）；返回负值按 0 截断（防御，正常不可达）。
        """
        self._active_decode_provider = provider

    # ------------------------------------------------------------ 原语 --
    def register(self, port_id: int, owner: str) -> None:
        """流发射时登记一条指向该端口的在途流。"""
        if not owner:
            raise HbmPortFlowError(
                "hbm port flow requires a non-empty owner")
        self._counts[port_id] = self._counts.get(port_id, 0) + 1
        self._owners.setdefault(owner, []).append(port_id)
        self.has_registrations = True

    def unregister(self, port_id: int, owner: str) -> None:
        """完成事件注销一条（fail-closed：双释放/未知归属即报）。"""
        ports = self._owners.get(owner)
        if not ports or port_id not in ports:
            raise HbmPortFlowError(
                "unregister on unregistered port flow {!r} for owner "
                "{!r}: double release".format(port_id, owner))
        ports.remove(port_id)
        if not ports:
            del self._owners[owner]
        self._decrement(port_id)

    def release_owner(self, owner: str) -> int:
        """注销该归属键的全部在册端口流（完成事件驱动）；返回条数。

        未知归属返回 0（幂等）——释放四点（drain/完成/逐切片/merge
        watch）对无登记 owner 合法空放，与 LinkFlowRegistry.release_owner
        同口径；细粒度双释放检测由 ``unregister`` 承担。
        """
        ports = self._owners.pop(owner, None)
        if not ports:
            return 0
        for port_id in ports:
            self._decrement(port_id)
        return len(ports)

    # ------------------------------------------------------------ 除数 --
    def divisor(self, port_id: int) -> int:
        """u_port（在册他流数；候选自身 +1 由计价侧显式加）。"""
        return self._active_streams(port_id) + self._counts.get(port_id, 0)

    def decomposition(self, port_id: int) -> tuple[int, int]:
        """F4 分解：(活跃 decode 消费流数, 在册传输流数)。"""
        return (
            self._active_streams(port_id),
            self._counts.get(port_id, 0))

    def snapshot(self) -> dict:
        """端口快照（决策日志/审计用；字段名与 C5 冻结的 port_snapshot
        逐实例分解字段同名，C11 步骤 6 接通时零改名）。"""
        ports = sorted(set(self._counts) | {
            port for ports in self._owners.values() for port in ports})
        return {
            port_id: {
                "u_port_active_decode_streams": self._active_streams(port_id),
                "u_port_registered_transfer_flows": (
                    self._counts.get(port_id, 0)),
                "u_port_total": self.divisor(port_id),
            }
            for port_id in ports}

    def leaked_owners(self) -> dict:
        """在册归属视图（漏释放审计：结算边界后仍非空 = 泄漏证据）。"""
        return {
            owner: tuple(ports)
            for owner, ports in sorted(self._owners.items())}

    # ------------------------------------------------------------ 内部 --
    def _active_streams(self, port_id: int) -> int:
        if self._active_decode_provider is None:
            return 0
        return max(0, int(self._active_decode_provider(port_id)))

    def _decrement(self, port_id: int) -> None:
        count = self._counts.get(port_id, 0) - 1
        if count > 0:
            self._counts[port_id] = count
        else:
            self._counts.pop(port_id, None)


class CandidateStageHbmShareError(ValueError):
    """fail-closed：候选阶段份额视图的非法占用计数。"""


class CandidateStageHbmShare:
    """D06（2026-09-27，v2.0 §9.1 式(6)）：单候选阶段的 HBM 端口份额
    视图（**预测专用、只读叠加**）。

    式(6)：同一端口上计算读取、copy 写入、pool 恢复写入必须出现在
    同一个分母中——μ_e = η_e·B_e/(m_e + k_e)，其中
    m_e = 在册后台流（活跃 decode + 在册传输，仍由
    ``HbmPortFlowRegistry`` 独家描述），k_e = **本候选阶段**实际并发
    的候选逻辑流数（计算读一条流——同一计算流先后读权重和 KV 不拆成
    多条流；传输腿按通道/缓冲协议的并发界计数，串行通道上的排队层组
    不重复计）。本候选临时占用**另加、不污染真实登记**：视图从不调用
    register/unregister，真实登记表仍只由执行事件驱动；阶段结束后只有
    本预测视图失效，真实状态不变。

    口径（与 ``HbmPortFlowRegistry.divisor`` 的"消费方显式 +1"约定
    同族）：

    * ``divisor(port)`` = m_e + (k_e − 1)——原样传给既有计价原语
      （``_shard_endpoint_divisors`` → ``_shard_leg_ns`` 的 u+1、
      ``_pool_transfer_ns_shards`` 的 u_port+1），每条候选流即得
      B/(m_e + k_e) 同款份额（k_e = 1 的端口退化为旧口径 m_e+1，
      零漂移）；
    * ``stage_denominator(port)`` = m_e + k_e——本地算子求值器的
      HBM 份额（``hbm_share_divisor``，roofline 带宽项同分母）；
    * 不在占用映射内的端口 k_e = 1（该端口上至多本候选的一条流）。

    端口占用映射 fail-closed：k_e 必须为正整数（一条在册候选流至少
    计 1），非法输入即抛 ``CandidateStageHbmShareError``。
    """

    def __init__(
        self,
        port_registry: Optional["HbmPortFlowRegistry"],
        candidate_flows_by_port: Optional[Mapping[int, int]] = None,
    ) -> None:
        self._registry = port_registry
        self._candidate: dict[int, int] = {}
        if candidate_flows_by_port is not None:
            for port_id, flows in candidate_flows_by_port.items():
                if isinstance(flows, bool) or not isinstance(flows, int):
                    raise CandidateStageHbmShareError(
                        "candidate stage flows must be int, got "
                        f"{flows!r} for port {port_id!r}")
                if flows < 1:
                    raise CandidateStageHbmShareError(
                        "candidate stage flows must be >= 1, got "
                        f"{flows} for port {port_id!r}")
                self._candidate[int(port_id)] = int(flows)

    # ------------------------------------------------------------ 查询 --
    def candidate_flows(self, port_id: int) -> int:
        """k_e：本候选阶段在该端口上的并发逻辑流数（缺省 1）。"""
        return self._candidate.get(int(port_id), 1)

    def registered_divisor(self, port_id: int) -> int:
        """m_e：在册后台流数（活跃 decode + 在册传输）——纯真实
        登记视图，与候选占用无关（未注入注册表 = 0，离线/单测口径）。"""
        if self._registry is None:
            return 0
        return max(0, int(self._registry.divisor(port_id)))

    def divisor(self, port_id: int) -> int:
        """m_e + (k_e − 1)——消费方显式 +1 口径（原语兼容位）。"""
        return self.registered_divisor(port_id) + (
            self.candidate_flows(port_id) - 1)

    def stage_denominator(self, port_id: int) -> int:
        """m_e + k_e——式(6) 完整分母（本地算子求值器 HBM 份额）。"""
        return self.registered_divisor(port_id) + self.candidate_flows(
            port_id)

    def snapshot(self) -> dict:
        """份额快照（决策日志/审计；含在册分解与候选占用，只读）。"""
        ports = set(self._candidate)
        if self._registry is not None:
            ports.update(int(port) for port in self._registry.snapshot())
        return {
            port_id: {
                "u_port_registered": self.registered_divisor(port_id),
                "candidate_stage_flows": self.candidate_flows(port_id),
                "stage_denominator": self.stage_denominator(port_id),
            }
            for port_id in sorted(ports)}
