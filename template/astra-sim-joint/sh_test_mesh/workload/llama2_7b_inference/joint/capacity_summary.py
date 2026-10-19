"""capacity_summary.py -- D11：分阶段足迹 + 容量驱逐摘要（v2.0 §14）。

设计依据：《联合代价模型》§14（式 19/20、§14.2 驱逐摘要数据结构、
§14.3 释放完成、§14.4 外部性不计价）与 joint修改文档 D11。

取代裁定（joint修改文档 §0.2 取代二）：E 进候选代价**不走候选内递推**
（懒式/多级精化明令禁止），改为**增量维护的容量摘要**进入估价——

* ``KVCacheManager``（face_scheduler）在完成/激活/引用变化/真实释放
  事件时增量维护摘要（``_refresh_capacity_summary``，挂在
  ``_check_invariants_after_mutation`` 咽喉点 + 阶段引用两入口）；候选
  **只查询**冻结快照（``capacity_summary_snapshot``），查询成本
  O(桶数×rank 数)，不随已完成 session 累计数增长；
* ``_ensure_capacity`` 的真实 T/E/victim 决策保留原位、不挪进候选循环
  ——摘要只是近似视图，实际提交仍按真实规则重新确认（§14.2）。

容量账（式 19）：C_r = W_r + O_r + Z_r + F_r 分量分明——W 常驻权重、
O 已物化实际占用、Z 其他事务尚未物化的承诺空间、F 可新预约空间；一次
分配从 Z 转 O 只转账、不改变 C−W 总量。峰值对准备/prefill/decode/merge
四个时间阶段分别检查（式 20 ``Δb_r = max(0, P^peak,new_r − F_r)``），
不机械相加成"全程同时驻留"。

驱逐摘要（§14.2）：按 T 类别 × E 易释放后缀/其余层 × 写回端口/路径
归属分桶；同一桶的释放原子具有同一种逐 rank 分片比例（比例不同必须
分桶——不能把本来耦合的 TP 分片拆成各 rank 独立选择 victim）。每桶
保存逐 rank 可释放向量 R、干净/脏向量 D、原子个数/最大原子向量。干净
表示释放已有可直接复用的有效 backing、不付池写回；脏表示释放前必须
写回（本仓执行协议：容量逐出恒经 remote_store 写回、账面忽略池侧陈旧
副本（face_scheduler §4.3.1），驻留字节无有效池 backing——clean 向量
由池 backing 交叠真实推导，本协议下恒空，结构保留、计价分支钉测）。

θ 份额近似消费（§14.2）：按固定桶序消费逐 rank 缺口向量 d。对桶 b 取
θ_b = min(1, max_{r:R_{b,r}>0} d_r / R_{b,r})；若该桶不能覆盖任何仍有
缺口的 rank 则 θ 为 0；预计释放 θ_b·R_{b,r} 并更新
d_r = max(0, d_r − θ_b·R_{b,r})；预计脏写回 θ_b·D_{b,r}，预测事务数
使用相同比例的原子数作启动费用近似。这是连续份额近似、不是真实释放
计划——实际提交仍只能释放完整、合法的对象/层组。

外部性（§14.4）：当前请求只承担其真实前驱上的写回、空间等待和自己需
要的历史恢复；被驱逐会话未来返回的恢复费用**不属于**本请求代价——
本模块不产生任何 victim 未来恢复计价项。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping, Optional, Sequence

#: 峰值检查的四个时间阶段（式 20：准备/prefill/decode/merge）。
PHASE_PREPARE = "prepare"
PHASE_PREFILL = "prefill"
PHASE_DECODE = "decode"
PHASE_MERGE = "merge"
PHASE_ORDER = (
    PHASE_PREPARE,
    PHASE_PREFILL,
    PHASE_DECODE,
    PHASE_MERGE,
)

#: E 的桶内层类：易释放后缀（E 软目标外的后缀区间，先释放）/其余层
#: （只能整份释放触达的前缀区间）。二者字节集不相交，并集 = 全部可释放
#: 驻留字节。
SUFFIX_EASY = "easy_suffix"
SUFFIX_REST = "rest_layers"
SUFFIX_CLASS_ORDER = (SUFFIX_EASY, SUFFIX_REST)

#: lru（T-off）模式的单一遍类别名（与 ``EvictionClassPass(None)`` 对应）。
CATEGORY_ALL = "all"

#: 摘要已登记误差来源标签（§5.5 "包含版本与误差来源"）。摘要是近似视
#: 图（§14.2），两个结构性误差源恒在册：
#: * theta_share_continuous_approximation——θ 份额是连续近似（非真实
#:   释放计划；实际提交只释放完整、合法的对象/层组）；
#: * incremental_maintenance_lag——摘要经事件侧增量维护，查询时点与
#:   最近维护事件之间存在滞后窗口。
CAPACITY_SUMMARY_ERROR_THETA_SHARE = "theta_share_continuous_approximation"
CAPACITY_SUMMARY_ERROR_MAINTENANCE_LAG = "incremental_maintenance_lag"
CAPACITY_SUMMARY_ERROR_SOURCES = (
    CAPACITY_SUMMARY_ERROR_THETA_SHARE,
    CAPACITY_SUMMARY_ERROR_MAINTENANCE_LAG,
)


class CapacitySummaryError(ValueError):
    """fail-closed：摘要构造或 θ 消费的非法输入。"""


def _check_rank_vector(name: str, values: Sequence[int], size: int) -> tuple[int, ...]:
    normalized = tuple(int(value) for value in values)
    if len(normalized) != size:
        raise CapacitySummaryError(
            f"{name} length {len(normalized)} does not match rank count {size}")
    if any(value < 0 for value in normalized):
        raise CapacitySummaryError(f"{name} contains a negative byte count")
    return normalized


def shard_ratio_signature(values: Sequence[int]) -> tuple[int, ...]:
    """逐 rank 分片比例签名（gcd 归一非零向量；全零向量返回全零）。

    KV 字节对 token 数**精确线性**（face_scheduler §7.4 计量注释钉），
    故 per-token 逐 rank 向量 = 实例内一切释放原子共享的同一比例；gcd
    归一使整数比例相同的向量得到同一签名——比例不同必须分桶（§14.2，
    不能把本来耦合的 TP 分片拆成各 rank 独立选择 victim）。
    """
    normalized = tuple(int(value) for value in values)
    nonzero = [value for value in normalized if value > 0]
    if not nonzero:
        return normalized
    divisor = math.gcd(*nonzero)
    return tuple(value // divisor for value in normalized)


@dataclass(frozen=True)
class PhaseFootprint:
    """候选分阶段峰值足迹（式 19/20；逐阶段逐 rank 的**新增**峰值字节）。

    四阶段峰值各自独立对照 F 检查（``binding_gap_by_rank``）；不同时间
    释放的缓冲和持久 KV 不机械相加成"全程同时驻留"（§14.1）。所有向
    量等长（= 实例 TP rank 数）、非负。
    """

    prepare: tuple[int, ...]
    prefill: tuple[int, ...]
    decode: tuple[int, ...]
    merge: tuple[int, ...]

    def __post_init__(self) -> None:
        sizes = {
            len(self.prepare),
            len(self.prefill),
            len(self.decode),
            len(self.merge),
        }
        if len(sizes) != 1:
            raise CapacitySummaryError(
                "PhaseFootprint phase vectors must share one rank count, "
                f"got prepare={len(self.prepare)}, prefill={len(self.prefill)}, "
                f"decode={len(self.decode)}, merge={len(self.merge)}")
        for name in PHASE_ORDER:
            vector = self.phase(name)
            if any(value < 0 for value in vector):
                raise CapacitySummaryError(
                    f"PhaseFootprint phase {name} contains a negative byte count")

    def phase(self, name: str) -> tuple[int, ...]:
        if name not in PHASE_ORDER:
            raise CapacitySummaryError(f"unknown footprint phase {name!r}")
        return getattr(self, name)

    def phases(self) -> tuple[tuple[str, tuple[int, ...]], ...]:
        return tuple((name, self.phase(name)) for name in PHASE_ORDER)

    def phase_gaps(
        self, free_by_rank: Sequence[int],
    ) -> tuple[tuple[str, tuple[int, ...]], ...]:
        """式(20) 逐阶段缺口 Δb_r = max(0, P^peak_r − F_r)。"""
        free = tuple(int(value) for value in free_by_rank)
        if len(free) != len(self.prepare):
            # (2026-10-01 审查修复) 自由向量长度与足迹 rank 数不一致 =
            # 调用方 bug——旧实现静默按 free=0（保守）或静默截断处理，
            # 与全仓 fail-closed 姿态相悖（同款 _check_rank_vector 均
            # raise）。
            raise CapacitySummaryError(
                f"free vector rank count {len(free)} != footprint rank "
                f"count {len(self.prepare)}")
        gaps = []
        for name, peak in self.phases():
            # (2026-10-01 R2 清理) 长度门已保证 free 与 peak 等长——
            # 旧的缺位按 0 条件分支不可达，直取 free[rank]。
            gaps.append((name, tuple(
                max(0, value - free[rank])
                for rank, value in enumerate(peak))))
        return tuple(gaps)

    def binding_gap_by_rank(
        self, free_by_rank: Sequence[int],
    ) -> tuple[int, ...]:
        """逐 rank 逐阶段缺口的逐 rank max（驱逐需求的绑定向量）。"""
        per_phase = [gap for _name, gap in self.phase_gaps(free_by_rank)]
        if not per_phase:
            return ()
        size = len(per_phase[0])
        return tuple(
            max(gap[rank] for gap in per_phase) for rank in range(size))

    def binding_phase(self, free_by_rank: Sequence[int]) -> str:
        """峰值最大的阶段名（审计披露：绑定阶段）。"""
        gaps = self.phase_gaps(free_by_rank)
        binding = PHASE_PREPARE
        binding_total = -1
        for name, gap in gaps:
            total = sum(gap)
            if total > binding_total:
                binding = name
                binding_total = total
        return binding

    @classmethod
    def single(
        cls, name: str, vector: Sequence[int],
    ) -> "PhaseFootprint":
        """单阶段足迹（merge 腿等容量准备语境；其余阶段恒零）。"""
        values = tuple(int(value) for value in vector)
        zeros = tuple(0 for _ in values)
        fields = {phase: zeros for phase in PHASE_ORDER}
        fields[name] = values
        return cls(**fields)


@dataclass(frozen=True)
class CapacityBucketKey:
    """桶键（§14.2 分桶三轴；任一轴不同即不同桶）。

    * ``category``：T 类别（typed 模式 human/tool，lru 模式单遍 all）；
    * ``suffix_class``：E 易释放后缀 / 其余层；
    * ``shard_ratio``：逐 rank 分片比例签名（比例不同必须分桶）；
    * ``writeback_ports``：逐 rank 写回端口（pool edge）签名——端口
      归属不同的释放原子不混桶（共享池端口竞争在计价侧按实际 edge
      逐 rank 汇流，见 joint_cost_model `_space_ready_ns_estimate`）。
    """

    category: str
    suffix_class: str
    shard_ratio: tuple[int, ...]
    writeback_ports: tuple[int, ...]

    def describe(self) -> str:
        return "{}/{}".format(self.category, self.suffix_class)


@dataclass(frozen=True)
class CapacityBucket:
    """单桶聚合（§14.2：摘要只做聚合估值，桶内仍按原有对象排序执行）。

    R = reclaimable_by_rank（逐 rank 总可释放）；clean + dirty == R
    （守恒钉测）；``atom_count_by_rank`` = 释放原子个数向量（本仓释放
    原子 = 单 victim 的单层类区间一次 store 事务，TP 耦合——各 rank
    分量相等）；``max_atom_by_rank`` = 最大单原子逐 rank 向量。
    """

    key: CapacityBucketKey
    reclaimable_by_rank: tuple[int, ...]
    clean_by_rank: tuple[int, ...]
    dirty_by_rank: tuple[int, ...]
    atom_count_by_rank: tuple[int, ...]
    max_atom_by_rank: tuple[int, ...]

    def __post_init__(self) -> None:
        size = len(self.reclaimable_by_rank)
        vectors = (
            ("clean_by_rank", self.clean_by_rank),
            ("dirty_by_rank", self.dirty_by_rank),
            ("atom_count_by_rank", self.atom_count_by_rank),
            ("max_atom_by_rank", self.max_atom_by_rank),
        )
        if self.key.shard_ratio and len(self.key.shard_ratio) != size:
            raise CapacitySummaryError(
                "bucket shard_ratio length does not match rank count")
        if self.key.writeback_ports and len(self.key.writeback_ports) != size:
            raise CapacitySummaryError(
                "bucket writeback_ports length does not match rank count")
        for name, values in vectors:
            _check_rank_vector(name, values, size)
        for rank, (reclaimable, clean, dirty) in enumerate(zip(
                self.reclaimable_by_rank,
                self.clean_by_rank,
                self.dirty_by_rank)):
            if clean + dirty != reclaimable:
                raise CapacitySummaryError(
                    f"bucket {self.key.describe()} rank {rank} violates "
                    f"clean+dirty==reclaimable ({clean}+{dirty}!={reclaimable})")
            if self.atom_count_by_rank[rank] == 0 and reclaimable > 0:
                raise CapacitySummaryError(
                    f"bucket {self.key.describe()} rank {rank} has "
                    f"reclaimable bytes but zero atoms")
            if reclaimable > self.max_atom_by_rank[rank] * self.atom_count_by_rank[rank]:
                raise CapacitySummaryError(
                    f"bucket {self.key.describe()} rank {rank} reclaimable "
                    "bytes exceed atom_count × max_atom")


@dataclass(frozen=True)
class InstanceCapacitySummary:
    """单实例摘要快照：固定桶序（T 类别序 × 层类序 × 键余序）。

    §5.5（2026-09-30 修复）：携带版本与误差来源——``snapshot_version``
    = 生产管理器累计维护事件数（KVCacheManager 每次增量维护 +1，快照
    时刻冻结）；``maintenance_event_count`` = 快照时刻的维护事件累计
    计数；``error_sources`` = 结构性误差来源标签（θ 份额连续近似 +
    增量维护滞后）。直构夹具/合成锚缺省 0/空——"无版本"是显式事实，
    不是未知冒充。
    """

    instance_index: int
    buckets: tuple[CapacityBucket, ...] = ()
    snapshot_version: int = 0
    maintenance_event_count: int = 0
    error_sources: tuple[str, ...] = ()

    @property
    def total_reclaimable_by_rank(self) -> tuple[int, ...]:
        if not self.buckets:
            return ()
        size = len(self.buckets[0].reclaimable_by_rank)
        totals = [0 for _ in range(size)]
        for bucket in self.buckets:
            for rank, value in enumerate(bucket.reclaimable_by_rank):
                totals[rank] += value
        return tuple(totals)


@dataclass(frozen=True)
class CapacitySummary:
    """全实例摘要（KVCacheManager 快照；候选只读）。

    §5.5（2026-09-30 修复）：``snapshot_version``/``error_sources`` 与
    各实例切片同源同值（管理器维护事件计数与登记误差源）。
    """

    instances: Mapping[int, InstanceCapacitySummary] = field(
        default_factory=dict)
    snapshot_version: int = 0
    error_sources: tuple[str, ...] = ()

    def instance(self, instance_index: int) -> InstanceCapacitySummary:
        """取单实例切片；缺席索引 = 零桶摘要（零可回收，合法保守态——
        生产管理器为全部拓扑实例建桶，缺席仅见于替身/离线视图）。"""
        summary = self.instances.get(instance_index)
        if summary is None:
            return InstanceCapacitySummary(
                instance_index=instance_index)
        return summary


@dataclass(frozen=True)
class BucketConsumption:
    """单桶 θ 消费结果（§14.2 连续份额近似；非真实释放计划）。"""

    key: CapacityBucketKey
    theta: float
    released_by_rank: tuple[int, ...]        # θ·R（缺口覆盖口径，cap 于 d）
    dirty_released_by_rank: tuple[int, ...]  # θ·dirty（计价口径，不 cap）
    clean_released_by_rank: tuple[int, ...]  # θ·clean（零写回费）
    expected_transactions: int               # ceil(θ·原子个数)——启动费用近似


def _proportional_bytes(
    theta: float, values: tuple[int, ...],
) -> tuple[int, ...]:
    """θ 份额逐 rank 字节（半单调整数化，cap 于原值）。"""
    scaled = []
    for value in values:
        approximated = int(theta * value + 0.5)
        scaled.append(min(value, max(0, approximated)))
    return tuple(scaled)


def consume_gap_theta(
    buckets: Sequence[CapacityBucket],
    gap_by_rank: Sequence[int],
) -> tuple[BucketConsumption, ...]:
    """按固定桶序以 θ 份额近似消费逐 rank 缺口（§14.2 原式）。

    * θ_b = min(1, max_{{r: R_b,r > 0}} d_r / R_b,r)；桶不能覆盖任何
      仍有缺口的 rank（全部 R_b,r == 0 于缺口 rank）⇒ θ = 0 跳过；
    * d_r = max(0, d_r − θ_b·R_b,r)（TP 耦合：θ 由最重缺口 rank 决定，
      其余 rank 同比例释放——不会各 rank 独立选择 victim）；
    * 预计脏写回 = θ_b·D_b,r；干净 = θ_b·clean_b,r（零写回费）；
      预测事务数 = ceil(θ_b·原子个数)（启动费用近似）。
    """
    d = [int(value) for value in gap_by_rank]
    if any(value < 0 for value in d):
        raise CapacitySummaryError("gap vector contains a negative byte count")
    consumptions: list[BucketConsumption] = []
    for bucket in buckets:
        if not any(d):
            break
        size = len(d)
        reclaimable = bucket.reclaimable_by_rank
        if len(reclaimable) != size:
            raise CapacitySummaryError(
                f"bucket {bucket.key.describe()} rank count {len(reclaimable)} "
                f"does not match gap vector length {size}")
        ratios = [
            d[rank] / reclaimable[rank]
            for rank in range(size)
            if reclaimable[rank] > 0 and d[rank] > 0
        ]
        if not ratios:
            continue
        theta = min(1.0, max(ratios))
        released = tuple(
            min(d[rank], _scaled)
            for rank, _scaled in enumerate(
                _proportional_bytes(theta, reclaimable)))
        dirty_released = _proportional_bytes(theta, bucket.dirty_by_rank)
        clean_released = _proportional_bytes(theta, bucket.clean_by_rank)
        atom_count = max(bucket.atom_count_by_rank, default=0)
        expected_transactions = int(math.ceil(theta * atom_count))
        for rank in range(size):
            d[rank] = max(0, d[rank] - released[rank])
        consumptions.append(BucketConsumption(
            key=bucket.key,
            theta=theta,
            released_by_rank=released,
            dirty_released_by_rank=dirty_released,
            clean_released_by_rank=clean_released,
            expected_transactions=expected_transactions,
        ))
    return tuple(consumptions)


def synthesise_summary_from_reclaimable(
    instance_index: int,
    reclaimable_by_rank: Sequence[int],
    writeback_ports: Optional[Sequence[int]] = None,
) -> InstanceCapacitySummary:
    """平坦可回收向量 → 单桶摘要（退化锚，v2.0 前测试夹具/离线视图）。

    直接构造 ``InstanceLoadView`` 而未注入增量维护摘要的语境（既有测
    试夹具/离线视图）：按"零分桶信息"退化为单 rest 桶（类别 all、整
    段脏、单原子）——θ 消费公式与生产路径同一份，仅分桶信息退化；
    生产路径（SH ``_joint_cost_model``）恒注入 KVCacheManager 快照。
    空/零长向量（缺省视图 = 无信息）返回零桶摘要（零可回收，保守）。
    """
    vector = _check_rank_vector(
        "reclaimable_by_rank", reclaimable_by_rank,
        len(reclaimable_by_rank))
    if not vector:
        return InstanceCapacitySummary(instance_index=instance_index)
    if writeback_ports is None:
        ports = tuple(0 for _ in vector)
    else:
        ports = tuple(int(value) for value in writeback_ports)
        if len(ports) != len(vector):
            raise CapacitySummaryError(
                "writeback_ports length does not match rank count")
    nonzero = any(value > 0 for value in vector)
    atoms = tuple(1 if value > 0 else 0 for value in vector)
    bucket = CapacityBucket(
        key=CapacityBucketKey(
            category=CATEGORY_ALL,
            suffix_class=SUFFIX_REST,
            shard_ratio=shard_ratio_signature(vector),
            writeback_ports=ports,
        ),
        reclaimable_by_rank=vector,
        clean_by_rank=tuple(0 for _ in vector),
        dirty_by_rank=vector,
        atom_count_by_rank=atoms if nonzero else tuple(0 for _ in vector),
        max_atom_by_rank=vector,
    )
    return InstanceCapacitySummary(
        instance_index=instance_index, buckets=(bucket,))
