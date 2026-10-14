#!/usr/bin/env python3
"""test_joint_shard_pricing.py -- C1（WP1a）逐 shard 三腿 min 计价 +
D1 读放大因果口径的零后端单测（F1 冻结形态的解析锚点）。

覆盖（卡 C1 测试清单）：
  1. 平价锚点：d ≤ ρ 空链路单流下 remote-read 每步代价 ≡ 本地读平价
     （t_r = max(t_l, c·d/B_D2D) 的 d≤ρ 分支退化为 t_l）——注意**仅
     历史扫描项平价，非整轮平价保证**（设计文档 §1.2：供给条件不能
     单独保证完整轮次与本地执行平价）；
  2. 三腿 min 三夹具（noc 快/HBM 慢、noc 慢/HBM 快、双慢）：断言取
     max 不取和；
  3. TP 并行：链路不相交路径 wall = 单 shard 时间（非 N× 串行放大）；
     共链时自身重叠计费保留（divisor 含 self）；零字节 shard 无流
     不争用；
  4. 读放大因果（D1）：decode 每步远读基数 = 仍位于远端的基础历史
     （H，非 H+I）；prefill 按实际消费遍数计（两遍扫描夹具遍数 = 2）；
  5. credit 逐 shard 切块：skewed rank 字节下 first/remaining 由最大
     rank 主导；K ≥ read_passes 单 credit 退化不变量；
  6. 代表路径广播锚（route_paths_fn 未注入）：字节均衡、空链路时
     与旧聚合口径（_transfer_ns）同 wall 值（C4 金值重推导的桥接锚）；
  7. 池腿逐 shard（D04，取代 F3 聚合口径）：copy 混合腿的池后缀按
     _pool_transfer_ns_shards 逐 rank 计价（式 7），数值回归钉死。

运行：cd sh_test_mesh/workload/llama2_7b_inference &&
      python3 joint/test_joint_shard_pricing.py   （或 pytest 同路径）
"""
import math
import os
import sys
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
for _p in (_HERE, _PARENT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from joint.joint_cost_model import (  # noqa: E402
    ACTION_COPY,
    ACTION_REMOTE,
    InstanceLoadView,
    JointCostError,
    JointCostModel,
    JointHardwareRates,
    LinkFlowRegistry,
    RequestView,
    ServiceFactors,
    SessionKVView,
    _pool_transfer_ns_shards,
    _transfer_ns,
    _transfer_ns_shards,
    _shard_stream_ns_shards,
)
from joint.hbm_port_flow_registry import HbmPortFlowRegistry  # noqa: E402


# ============================================================ 夹具构造 ==

def _rates(noc=10.0, hbm=100.0, lat=10, pool=5.0):
    return JointHardwareRates.from_gbps(
        noc_link_gbps=noc, pool_port_gbps=pool, local_hbm_gbps=hbm,
        d2d_latency_ns=lat, pool_latency_ns=100)


def _load(index=0, total=0):
    return InstanceLoadView(
        instance_index=index, queued_task_load_ns=total,
        running_task_load_ns=0, active_decode_task_load_ns=0,
        hbm_remaining_bytes_by_tp_rank=(10**9, 10**9))


def _session(home=0, resident=0, history_bytes=(1000, 1000),
             missing=(0, 0), location="local_hbm", prefix=4,
             history_tokens=100):
    return SessionKVView(
        session_id="s", home_instance=home, resident_instance=resident,
        location=location, history_tokens=history_tokens,
        resident_prefix_layers=prefix,
        history_bytes_by_tp_rank=history_bytes,
        missing_bytes_by_tp_rank=missing)


def _request(input_tokens=50, decode=10, input_bytes=(25, 25),
             prefill_scan_passes=None):
    return RequestView(
        request_id="r", session_id="s", input_tokens=input_tokens,
        history_tokens_before=100, estimated_decode_tokens=decode,
        horizon_source="session_online_mean",
        input_kv_bytes_by_tp_rank=input_bytes,
        prefill_scan_passes=prefill_scan_passes)


def _disjoint_paths(source, target):
    """tp=2 逐 rank 不相交路径（rank 对 (2s,2t)/(2s+1,2t+1)，各 1 跳）。"""
    return ((2 * source, 2 * target), (2 * source + 1, 2 * target + 1))


class PerRankFootprintTest(unittest.TestCase):
    """Capacity footprints preserve the KV ledger's physical TP partition."""

    def test_copy_footprint_keeps_skewed_rank_bytes(self):
        model = _model()
        session = _session(
            history_bytes=(100, 900), missing=(0, 0), resident=0)
        request = _request(input_bytes=(7, 13))
        self.assertEqual(
            model._space_footprint_by_tp_rank(
                session, request, instance_index=1, action=ACTION_COPY),
            (107, 913))

    def test_footprint_rejects_tp_vector_length_mismatch(self):
        model = _model()
        session = _session(history_bytes=(100, 900))
        request = _request(input_bytes=(7,))
        with self.assertRaisesRegex(JointCostError, "length"):
            model._space_footprint_by_tp_rank(
                session, request, instance_index=1, action=ACTION_COPY)


class CopyStreamingCostTest(unittest.TestCase):
    """Copy 递推：chunk 只门控其层区间，末 chunk 到达后的计算由式(13)
    递推保留（D02——旧 max(tail, compute) 吸收形态废除）。"""

    def test_copy_prefix_tail_overlaps_compute(self):
        model = _model(rates=_rates(noc=10.0, hbm=100.0, lat=0))
        model.model_layers = 16
        candidate = model.estimate_P_with_continuation(
            session=_session(
                history_bytes=(1600, 1600), prefix=16, resident=0),
            request=_request(input_tokens=50, decode=10),
            instance_index=1, action=ACTION_COPY, remote_enabled=True)
        self.assertTrue(candidate.applicable)
        # 审计 breakdown 保留全前缀传输量（160 ns = 逐 shard 1600/10，
        # disjoint 路径除数 1）。D02 递推：chunk 0（层 0-7）到达 80、
        # chunk 1（层 8-15）到达 160（通道内串行链 80+80）；prefill 计
        # 算 50 按层均分（2×4 + 14×3）：F7 = 88+3×6 = 106 → F8 =
        # max(106,160)+3 = 163 → F15 = 163+21 = 184（末 chunk 到达后
        # 的 8 层计算逐层保留——旧 max(80, 50) 形态整体吸收）；decode
        # 20 串行其后 ⇒ cost = 184 + 20 = 204。
        self.assertEqual(candidate.breakdown.history_prep_ns, 160)
        self.assertIn("prefill_first_forward_ns=184",
                      candidate.breakdown.notes)
        self.assertEqual(candidate.cost_ns, 204)
        self.assertIn("copy_handoff_chunks=2", candidate.breakdown.notes)
        self.assertIn("copy_handoff_first_chunk_ns=80",
                      candidate.breakdown.notes)
        self.assertIn("copy_handoff_chunk_walls=80,160",
                      candidate.breakdown.notes)


def _route(source, target):
    return ((source, target), 1)


def _model(loads=None, *, rates=None, route_paths_fn="disjoint",
           tile_bytes="auto"):
    return JointCostModel(
        rates=rates if rates is not None else _rates(),
        loads=loads if loads is not None else {0: _load(0), 1: _load(1)},
        flow_registry=LinkFlowRegistry(),
        service_factors=ServiceFactors(),
        prefill_ns_per_token=1.0, decode_ns_per_token=2.0,
        model_layers=4, instance_tp_size=2, route_fn=_route,
        route_paths_fn=(
            _disjoint_paths if route_paths_fn == "disjoint"
            else route_paths_fn),
        remote_read_tile_bytes=tile_bytes)


# ==================================================== 1. 平价锚点（§1.2）==


class ParityAnchorTest(unittest.TestCase):
    """d ≤ ρ 空链路单流：remote-read 每步历史扫描项 ≡ 本地读平价。"""

    def test_d_le_rho_degrades_to_local_scan(self):
        # 基座锚点：B_D2D=4050、B_HBM=1640 → ρ≈2.47；d=2 ≤ ρ、空链路、
        # 单流。三腿 max = max(c/4050, c/1640) = c/1640 = t_l（本地扫描
        # 项）——noc 腿（404.9）被 home 读腿（1000）遮蔽，与解析锚
        # t_r = max(t_l, c·d/B_D2D) 的 d≤ρ 分支退化为 t_l 同结论。
        # 注意：仅历史扫描项平价，非整轮平价保证（设计文档 §1.2——
        # 首块启动/流水填充/容量/同步另计，供给条件不单独保证整轮平价）。
        rates = _rates(noc=4050.0, hbm=1640.0, lat=5)
        shard_bytes = 1_640_000
        wall = _transfer_ns_shards(
            rates, LinkFlowRegistry(), None,
            paths_by_rank=(((0, 1, 2),),),
            bytes_by_rank=(shard_bytes,),
            include_self=True, kind="read")
        local_scan_ns = int(shard_bytes / rates.local_hbm_bytes_per_ns)
        self.assertEqual(local_scan_ns, 1000)
        self.assertLess(
            int(shard_bytes / rates.noc_link_bytes_per_ns), local_scan_ns)
        # 跳时延剥离后 ≡ 本地扫描项（hop 启动另计，不进扫描项平价）。
        self.assertEqual(wall - 2 * rates.d2d_latency_ns, local_scan_ns)

    def test_estimate_p_with_continuation_per_step_parity(self):
        # D06 修订（2026-09-27，v2.0 §9.1 式(6)）：本锚原断言"暂存写腿
        # ≡ 本地扫描项"（执行端写腿独占端口的旧口径）。D06 起执行端口
        # 上候选计算读与 remote 暂存写同分母（候选阶段 k = 计算读 +
        # 暂存写 = 2）——写腿 = 1640000×4/(1640/2) = 8000 主导（home
        # 读腿 4000、noc 腿 1619 均被遮蔽），全流 wall = 1×5 跳时延 +
        # 8000 = 8005。per-step 旧平价由"独占读平价"改为"共享份额"：
        # 写腿每步 = 1640000/820 = 2000 = 本地扫描项 × 候选阶段流数。
        rates = _rates(noc=4050.0, hbm=1640.0, lat=5)
        model = _model(rates=rates)
        candidate = model.estimate_P_with_continuation(
            session=_session(history_bytes=(1_640_000, 1_640_000),
                             history_tokens=100),
            request=_request(decode=3), instance_index=1,
            action=ACTION_REMOTE, remote_enabled=True)
        self.assertTrue(candidate.applicable)
        self.assertIn("remote_read_passes=4", candidate.breakdown.notes)
        self.assertEqual(
            candidate.breakdown.remote_read_ns,
            rates.d2d_latency_ns
            + (1_640_000 * 4) // (1640 // 2))
        # 共享份额逐步守恒：per-step 写腿 = 独占扫描 × k。
        self.assertEqual(
            (candidate.breakdown.remote_read_ns
             - rates.d2d_latency_ns) // 4,
            2 * (1_640_000 // 1640))

    def test_sub_parity_config_noc_leg_degrades(self):
        # 对照：ρ<1 配置（noc=1200 < hbm=1640、d=1 > ρ=0.73）——noc 腿
        # 主导，扫描项高于本地（锚点的 d>ρ 方向；不排除系统级收益，
        # 仅扫描项口径）。
        rates = _rates(noc=1200.0, hbm=1640.0, lat=5)
        shard_bytes = 1_640_000
        wall = _transfer_ns_shards(
            rates, LinkFlowRegistry(), None,
            paths_by_rank=(((0, 1),),),
            bytes_by_rank=(shard_bytes,),
            include_self=True, kind="read")
        # D17：腿服务时间向上取整（1_640_000/1200 = 1366.67 → 1367）。
        self.assertEqual(
            wall - rates.d2d_latency_ns, math.ceil(shard_bytes / 1200))
        self.assertGreater(
            wall - rates.d2d_latency_ns,
            math.ceil(shard_bytes / rates.local_hbm_bytes_per_ns))


# ==================================================== 2. 三腿 min 夹具 ==


class ThreeLegMaxTest(unittest.TestCase):
    """三腿 max 不取和：noc 快/HBM 慢、noc 慢/HBM 快、双慢三夹具。

    注：u_home/u_exec 本卡恒 0（port_registry=None，C2 接线后生效），
    copy 的 home/exec 两端点腿同率；三腿区分度由 noc 腿与端点腿给出。
    """

    BYTES = 6000

    def _wall(self, rates):
        return _transfer_ns_shards(
            rates, LinkFlowRegistry(), None,
            paths_by_rank=(((0, 1),),),
            bytes_by_rank=(self.BYTES,),
            include_self=True, kind="copy")

    def test_noc_fast_hbm_slow(self):
        # noc 腿 6 ns、home/exec 腿各 600 ns → max=600（和=1206）。
        rates = _rates(noc=1000.0, hbm=10.0, lat=0)
        noc_ns = self.BYTES / 1000.0
        hbm_ns = self.BYTES / 10.0
        self.assertEqual(self._wall(rates), int(max(noc_ns, hbm_ns)))
        self.assertNotEqual(self._wall(rates), int(noc_ns + 2 * hbm_ns))

    def test_noc_slow_hbm_fast(self):
        # noc 腿 6000 ns、端点腿各 6 ns → max=6000（和=6012）。
        rates = _rates(noc=1.0, hbm=1000.0, lat=0)
        self.assertEqual(self._wall(rates), self.BYTES)
        self.assertNotEqual(self._wall(rates), self.BYTES + 12)

    def test_both_slow(self):
        # noc 腿 3000、端点腿各 2000 → max=3000（和=7000）。
        rates = _rates(noc=2.0, hbm=3.0, lat=0)
        self.assertEqual(self._wall(rates), 3000)
        self.assertNotEqual(self._wall(rates), 7000)

    def test_read_kind_three_legs_exec_write_leg_at_u0(self):
        # C2 移交（C4 落地，2026-09-22）：A4' 补价 rider 后 kind="read"
        # 恒三腿（增执行端 HBM 写腿——镜像执行侧 remote-read 每 credit
        # 块到达的 COMM_WRITE 服务；F14 容量半边不动：读流直达消费、
        # 在途字节不占执行端 HBM 容量）。本夹具 u_home=u_exec=0
        # （port_registry=None 离线口径）：exec 写腿与 home 读腿同率
        # （各 6000/10 = 600），max 腿不变 → read 与 copy 同值 600
        #（断言锚不漂——C1 交付时两动作同值的成因自"read 无写腿"变为
        # "u=0 时写腿与读腿同值"）；kind 枚举外的值 fail-closed。
        rates = _rates(noc=1000.0, hbm=10.0, lat=0)
        read_wall = _transfer_ns_shards(
            rates, LinkFlowRegistry(), None,
            paths_by_rank=(((0, 1),),), bytes_by_rank=(self.BYTES,),
            include_self=True, kind="read")
        self.assertEqual(read_wall, 600)
        copy_wall = _transfer_ns_shards(
            rates, LinkFlowRegistry(), None,
            paths_by_rank=(((0, 1),),), bytes_by_rank=(self.BYTES,),
            include_self=True, kind="copy")
        self.assertEqual(read_wall, copy_wall)
        with self.assertRaises(Exception):
            _transfer_ns_shards(
                rates, LinkFlowRegistry(), None,
                paths_by_rank=(((0, 1),),), bytes_by_rank=(self.BYTES,),
                include_self=True, kind="prefetch")


# ============================================ 2.5 merge 目标写腿（D05）==


class MergeTargetWriteLegTest(unittest.TestCase):
    """merge 目标 HBM 写腿（D05，v2.0 C02/§15.2）：kind="merge" 三腿 =
    源读 + NoC + 落点写——与执行侧 noc_migrate 双端点登记同腿型（旧
    "merge 无 exec 写腿"计价废除；目标端口 = 各自方向路径末端 rank）。

    验收（joint修改文档 D05）：目标 HBM 慢 NoC 快时 merge 反映目标写
    瓶颈；容量充足不令该项消失（落点空间准备是另一前驱，仍在
    _space_available_ns_estimate 单列计价，与本写带宽腿正交）。
    """

    BYTES = 6000

    def _registry(self, contended_port, flows):
        registry = HbmPortFlowRegistry()
        registry.attach_active_decode_provider(
            lambda port: flows if port == contended_port else 0)
        return registry

    def test_target_write_bottleneck_when_target_hbm_slow_noc_fast(self):
        # 目标端口 rank1 在册 3 流（u_exec=3）、源端口 rank0 闲置：
        # legs = noc 6000/1000=6 / 源读 6000/10=600 / 目标写
        # 6000/(10/(3+1))=2400 → wall = 2400（写腿绑定）。旧无写腿口径
        # = max(6, 600) = 600——目标 HBM 慢（10）NoC 快（1000）时系统
        # 性漏计 4×目标写瓶颈。
        rates = _rates(noc=1000.0, hbm=10.0, lat=0)
        wall = _transfer_ns_shards(
            rates, LinkFlowRegistry(), self._registry(1, 3),
            paths_by_rank=(((0, 1),),), bytes_by_rank=(self.BYTES,),
            include_self=True, kind="merge")
        self.assertEqual(wall, 2400)
        self.assertNotEqual(wall, 600)

    def test_write_leg_registry_none_degenerates_to_read_parity(self):
        # port_registry=None（离线/单测口径）：u_home=u_exec=0 → 写腿与
        # 源读腿同值——merge ≡ copy = 600（D05 前既有金样未漂的成因：
        # hbm 快夹具下写腿被 noc/读腿遮蔽；hbm 慢夹具下同率腿不改 max）。
        rates = _rates(noc=1000.0, hbm=10.0, lat=0)
        merge_wall = _transfer_ns_shards(
            rates, LinkFlowRegistry(), None,
            paths_by_rank=(((0, 1),),), bytes_by_rank=(self.BYTES,),
            include_self=True, kind="merge")
        copy_wall = _transfer_ns_shards(
            rates, LinkFlowRegistry(), None,
            paths_by_rank=(((0, 1),),), bytes_by_rank=(self.BYTES,),
            include_self=True, kind="copy")
        self.assertEqual(merge_wall, 600)
        self.assertEqual(merge_wall, copy_wall)

    def test_estimate_merge_reflects_target_write_with_ample_capacity(self):
        # estimate_P_with_continuation 层（REMOTE 基，tp=2 disjoint）：
        # exec 增量 (12000,12000) ≤ home 保留 (50000,50000) → 前向
        # （byte_le 含等号）。前向路径 (2,0)/(3,1)：源端口 rank2/3 闲置、
        # 目标端口 rank0/1 活跃 decode 2 流 → 前向 = max(noc 12, 源读
        # 1200, 目标写 12000/(10/3)=3600) = 3600（目标写瓶颈，NoC 快
        # 1000B/ns 仅 12）；反向路径 (0,2)/(1,3)：源端口 rank0/1 u=2 →
        # 反向 = max(noc 50, 源读 15000, 目标写 5000) = 15000（仅披露）。
        # 双侧负载 hbm_remaining=10^9 ≫ 需求 → 空间准备为 0（无
        # home/exec_merge_eviction_writeback_est 注记）——容量充足不令
        # 目标写腿消失（该腿是带宽服务，不是空间等待）。
        rates = _rates(noc=1000.0, hbm=10.0, lat=0)
        registry = HbmPortFlowRegistry()
        registry.attach_active_decode_provider(
            lambda port: 2 if port in (0, 1) else 0)
        model = JointCostModel(
            rates=rates,
            loads={0: _load(0), 1: _load(1)},
            flow_registry=LinkFlowRegistry(),
            service_factors=ServiceFactors(),
            prefill_ns_per_token=1.0, decode_ns_per_token=2.0,
            model_layers=4, instance_tp_size=2, route_fn=_route,
            route_paths_fn=_disjoint_paths,
            hbm_port_registry=registry)
        candidate = model.estimate_P_with_continuation(
            session=_session(history_bytes=(50_000, 50_000)),
            request=_request(input_tokens=50, decode=10,
                             input_bytes=(10_000, 10_000)),
            instance_index=1, action=ACTION_REMOTE, remote_enabled=True)
        self.assertTrue(candidate.applicable)
        breakdown = candidate.breakdown
        self.assertIn("merge_v2_direction=forward", breakdown.notes)
        self.assertEqual(breakdown.merge_ns, 3600)
        self.assertIn("merge_v2_forward_ns=3600", breakdown.notes)
        self.assertIn("merge_v2_reverse_ns=15000", breakdown.notes)
        self.assertNotIn("home_merge_eviction_writeback_est",
                         breakdown.notes)
        self.assertNotIn("exec_merge_eviction_writeback_est",
                         breakdown.notes)


# ==================================================== 3. TP 并行口径 ==


class TpParallelismTest(unittest.TestCase):
    """TP6 不相交路径 wall = 单 shard 时间；共链自身重叠计费保留。"""

    BYTES_PER_RANK = 60

    def test_disjoint_paths_wall_is_single_shard(self):
        # 不相交：除数并集每链仅 1 条自身流 → divisor=1，各 shard 6 ns
        # 并行 → wall = 6 = 单 shard 时间；旧聚合口径（总 360 B ÷ 单
        # 链路率 10）= 36 = 6×（串行放大，即本卡废除的口径）。
        rates = _rates(noc=10.0, hbm=1e9, lat=0)
        paths = tuple(((rank, rank + 6),) for rank in range(6))
        bytes6 = (self.BYTES_PER_RANK,) * 6
        wall = _transfer_ns_shards(
            rates, LinkFlowRegistry(), None,
            paths_by_rank=paths, bytes_by_rank=bytes6,
            include_self=True, kind="read")
        self.assertEqual(wall, 6)
        legacy = _transfer_ns(
            total_bytes=sum(bytes6), path_hops=1, divisor=1,
            rates=rates, per_hop_latency_ns=None, startup_ns=0)
        self.assertEqual(legacy, 36)
        self.assertNotEqual(wall, legacy)

    def test_shared_link_keeps_self_overlap_billing(self):
        # 共链（6 条 TP 流同走 (0,6)）：divisor_multi 自身份额 = 6（divisor
        # 含 self）→ 每 shard 60/(10/6) = 36 ns → wall = 36 = 6× 单 shard
        # ——共享争用如实计费（自身重叠计费保留，divisor_multi 语义不动）。
        rates = _rates(noc=10.0, hbm=1e9, lat=0)
        registry = LinkFlowRegistry()
        paths = tuple(((0, 6),) for _ in range(6))
        bytes6 = (self.BYTES_PER_RANK,) * 6
        self.assertEqual(
            registry.divisor_multi(
                [path for rank_paths in paths for path in rank_paths],
                include_self=True), 6)
        wall = _transfer_ns_shards(
            rates, registry, None,
            paths_by_rank=paths, bytes_by_rank=bytes6,
            include_self=True, kind="read")
        self.assertEqual(wall, 36)
        # include_self=False 时不计自身流（除数回落 1）——参数语义对照。
        wall_no_self = _transfer_ns_shards(
            rates, registry, None,
            paths_by_rank=paths, bytes_by_rank=bytes6,
            include_self=False, kind="read")
        self.assertEqual(wall_no_self, 6)

    def test_zero_byte_shard_streams_nothing(self):
        # 零字节 shard 无流、不进除数并集：单 6000 B shard + 零字节
        # shard 共链 → wall = 600（若误计零字节自身份额则为 1200）。
        rates = _rates(noc=10.0, hbm=1e9, lat=0)
        wall = _transfer_ns_shards(
            rates, LinkFlowRegistry(), None,
            paths_by_rank=(((0, 6),), ((0, 6),)),
            bytes_by_rank=(6000, 0),
            include_self=True, kind="read")
        self.assertEqual(wall, 600)

    def test_copy_disjoint_through_estimate_p_with_continuation(self):
        # estimate_P_with_continuation 层（copy NoC 前缀腿）：disjoint 逐 rank 路径
        # → 每 rank 3000/10 = 300 → history_prep = 10 + 300；旧聚合口径
        # = 10 + 600（总 6000 ÷ 10）。
        model = _model(rates=_rates(noc=10.0, hbm=1e9))
        candidate = model.estimate_P_with_continuation(
            session=_session(history_bytes=(3000, 3000)),
            request=_request(), instance_index=1,
            action=ACTION_COPY, remote_enabled=True)
        self.assertTrue(candidate.applicable)
        self.assertEqual(candidate.breakdown.history_prep_ns, 10 + 300)

    def test_endpoint_leg_binds_through_estimate_p_with_continuation(self):
        # 端点腿正式消费（E12）+ D06 阶段份额（2026-09-27，式 6）：
        # noc=4050/hbm=1640 下 copy 候选执行端口并发流 k = 计算读 +
        # copy 留存写 = 2——exec 写腿 = 820000/(1640/2) = 1000 主导
        # （home 读腿 820000/1640 = 500、noc 腿 202.5 均被遮蔽）→
        # history_prep = 10 + 1000 > 旧聚合口径 10 + 405（D17：旧聚合
        # 原语带宽服务向上取整，1_640_000/4050 = 404.94）；写腿与计算
        # 读同分母、不各自独占整端口。
        model = _model(rates=_rates(noc=4050.0, hbm=1640.0))
        candidate = model.estimate_P_with_continuation(
            session=_session(history_bytes=(820_000, 820_000)),
            request=_request(), instance_index=1,
            action=ACTION_COPY, remote_enabled=True)
        self.assertEqual(candidate.breakdown.history_prep_ns, 10 + 1000)
        legacy = _transfer_ns(
            total_bytes=1_640_000, path_hops=1, divisor=1,
            rates=model.rates, per_hop_latency_ns=None, startup_ns=0)
        self.assertEqual(legacy, 10 + 405)
        self.assertGreater(candidate.breakdown.history_prep_ns, legacy)


# ==================================================== 4. 读放大因果（D1）==


class ReadAmplificationCausalTest(unittest.TestCase):
    """H=100k 基础 + I=20k 增量：decode 远读基数 = 100k（非 120k）。"""

    def _model(self):
        # noc=1000、hbm=1e9、lat=0、tp=2 disjoint → 纯 noc 腿可读数：
        # per-step per-rank = base_rank/1000。
        return _model(rates=_rates(noc=1000.0, hbm=1e9, lat=0))

    def test_decode_remote_base_is_history_only(self):
        # H = (50000, 50000) = 100k；I = (10000, 10000) = 20k（input +
        # 因果 decode 增长均属执行端本地读取，不计远读）。passes = 1
        # （prefill 单遍缺省）+ 10（decode 步）= 11 → 全流 wall = 50000
        # × 11 / 1000 = 550。若误用 H+I = 120k 基数则为 660。
        model = self._model()
        candidate = model.estimate_P_with_continuation(
            session=_session(history_bytes=(50_000, 50_000)),
            request=_request(input_tokens=50, decode=10,
                             input_bytes=(10_000, 10_000)),
            instance_index=1, action=ACTION_REMOTE, remote_enabled=True)
        self.assertTrue(candidate.applicable)
        self.assertEqual(candidate.breakdown.remote_read_ns, 550)
        self.assertNotEqual(candidate.breakdown.remote_read_ns, 660)
        self.assertIn("remote_read_passes=11", candidate.breakdown.notes)
        self.assertIn("prefill_scan_passes=1", candidate.breakdown.notes)

    def test_increment_size_does_not_inflate_remote_stream(self):
        # 同 H、I 放大 5×（100k 增量）：远读流不变（增量本地读取）。
        model = self._model()
        small = model.estimate_P_with_continuation(
            session=_session(history_bytes=(50_000, 50_000)),
            request=_request(input_bytes=(10_000, 10_000)),
            instance_index=1, action=ACTION_REMOTE, remote_enabled=True)
        large = model.estimate_P_with_continuation(
            session=_session(history_bytes=(50_000, 50_000)),
            request=_request(input_tokens=250, decode=10,
                             input_bytes=(50_000, 50_000)),
            instance_index=1, action=ACTION_REMOTE, remote_enabled=True)
        self.assertEqual(small.breakdown.remote_read_ns,
                         large.breakdown.remote_read_ns)

    def test_prefill_two_scan_passes_equals_two(self):
        # prefill 两遍扫描夹具：prefill_scan_passes=2、decode=0 → 遍数
        # = 2（prefill 对基础历史的实际消费遍数，非固定 1 遍）。
        model = self._model()
        candidate = model.estimate_P_with_continuation(
            session=_session(history_bytes=(50_000, 50_000)),
            request=_request(input_tokens=50, decode=0,
                             input_bytes=(10_000, 10_000),
                             prefill_scan_passes=2),
            instance_index=1, action=ACTION_REMOTE, remote_enabled=True)
        self.assertTrue(candidate.applicable)
        self.assertIn("prefill_scan_passes=2", candidate.breakdown.notes)
        self.assertIn("remote_read_passes=2", candidate.breakdown.notes)
        # 两遍 × 50k/1000/rank → 100 ns。
        self.assertEqual(candidate.breakdown.remote_read_ns, 100)

    def test_default_prefill_scan_derivation(self):
        # 因果缺省：input > 0 → 单遍；input == 0 且 decode == 0 → 无读流
        #（遍数 0，credit 拆分恒 0、无 K 披露）。
        model = self._model()
        candidate = model.estimate_P_with_continuation(
            session=_session(history_bytes=(50_000, 50_000)),
            request=_request(input_tokens=50, decode=0,
                             input_bytes=(10_000, 10_000)),
            instance_index=1, action=ACTION_REMOTE, remote_enabled=True)
        self.assertIn("remote_read_passes=1", candidate.breakdown.notes)
        empty = model.estimate_P_with_continuation(
            session=_session(history_bytes=(50_000, 50_000)),
            request=_request(input_tokens=0, decode=0, input_bytes=(0, 0)),
            instance_index=1, action=ACTION_REMOTE, remote_enabled=True)
        self.assertTrue(empty.applicable)
        self.assertEqual(empty.breakdown.remote_read_ns, 0)
        self.assertEqual(empty.breakdown.remote_read_first_credit_ns, 0)
        self.assertEqual(empty.breakdown.remote_read_stream_ns, 0)

    def test_merge_legs_per_shard_disjoint(self):
        # merge 双腿逐 shard（disjoint）：exec 增量 (12000,12000)（input
        # (10000,10000) + 增长 (2000,2000)）→ 前向 = 12000/1000 = 12；
        # home 保留 (50000,50000) → 反向 = 50 → merge = min = 12（旧
        # 聚合前向 = 24000/1000 = 24，TP 并行减半）。
        model = self._model()
        candidate = model.estimate_P_with_continuation(
            session=_session(history_bytes=(50_000, 50_000)),
            request=_request(input_tokens=50, decode=10,
                             input_bytes=(10_000, 10_000)),
            instance_index=1, action=ACTION_REMOTE, remote_enabled=True)
        self.assertEqual(candidate.breakdown.merge_ns, 12)
        self.assertIn("merge_v2_forward_ns=12", candidate.breakdown.notes)


# ==================================================== 5. credit 逐 shard ==


class CreditShardSplitTest(unittest.TestCase):
    """D09 字节 tile 审计拆分：首 tile / 其余流送段逐 shard 取 max
    （tile = 配置 auto = 256 token × 100 B/token = 25600 B/shard）。"""

    def test_skewed_ranks_driven_by_max_rank(self):
        # history (90000, 10000)、passes = 1+3 = 4、tile = 25600：
        # 首 tile = (25600, 25600) → 25；其余 = (334400, 14400) →
        # max(334, 14) = 334；全流审计 = max(360, 40) = 360（不变）。
        rates = _rates(noc=1000.0, hbm=1e9, lat=0)
        model = _model(rates=rates)
        candidate = model.estimate_P_with_continuation(
            session=_session(history_bytes=(90_000, 10_000)),
            request=_request(input_tokens=20, decode=3,
                             input_bytes=(2000, 2000)),
            instance_index=1, action=ACTION_REMOTE, remote_enabled=True)
        breakdown = candidate.breakdown
        self.assertIn("remote_read_tile_tokens=256", breakdown.notes)
        tile_bytes = 25_600
        paths = (_disjoint_paths(0, 1),)
        paths_by_rank = tuple((path,) for path in paths[0])
        expected_first = _transfer_ns_shards(
            rates, LinkFlowRegistry(), None,
            paths_by_rank=paths_by_rank,
            bytes_by_rank=(tile_bytes, tile_bytes),
            include_self=True, kind="read")
        expected_stream = _shard_stream_ns_shards(
            rates, LinkFlowRegistry(), None,
            paths_by_rank=paths_by_rank,
            bytes_by_rank=(360_000 - tile_bytes, 40_000 - tile_bytes),
            include_self=True, kind="read")
        self.assertEqual(breakdown.remote_read_first_credit_ns,
                         expected_first)
        self.assertGreater(expected_first, 0)
        self.assertEqual(breakdown.remote_read_stream_ns, expected_stream)
        # D17：流送腿向上取整——334_400/1000 = 334.4 → 335。
        self.assertEqual(breakdown.remote_read_stream_ns, 335)
        self.assertEqual(breakdown.remote_read_ns, 360)

    def test_single_tile_stream_degenerate_invariant(self):
        # 流字节 ≤ 一个 tile（5000×4=20000 ≤ 25600）：首 tile = 全流、
        # remaining == 0（单 tile 退化不变量在 tile 口径下保持）。
        model = _model(
            rates=_rates(noc=1000.0, hbm=1e9, lat=0), tile_bytes="auto")
        candidate = model.estimate_P_with_continuation(
            session=_session(history_bytes=(5_000, 5_000)),
            request=_request(input_tokens=20, decode=3,
                             input_bytes=(2000, 2000)),
            instance_index=1, action=ACTION_REMOTE, remote_enabled=True)
        breakdown = candidate.breakdown
        self.assertEqual(
            breakdown.remote_read_first_credit_ns,
            breakdown.remote_read_ns)
        self.assertEqual(breakdown.remote_read_stream_ns, 0)

    def test_hop_latency_only_in_first_credit(self):
        # 逐跳时延只进首 tile 暴露段（其余读流纯流送段无时延）：lat=10、
        # disjoint 1 跳 → first = 10 + ⌈25.6⌉ = 36、stream = 335（无时
        # 延；D17 腿服务向上取整）。
        rates = _rates(noc=1000.0, hbm=1e9, lat=10)
        model = _model(rates=rates)
        candidate = model.estimate_P_with_continuation(
            session=_session(history_bytes=(90_000, 10_000)),
            request=_request(input_tokens=20, decode=3,
                             input_bytes=(2000, 2000)),
            instance_index=1, action=ACTION_REMOTE, remote_enabled=True)
        breakdown = candidate.breakdown
        self.assertEqual(breakdown.remote_read_first_credit_ns, 26 + 10)
        self.assertEqual(breakdown.remote_read_stream_ns, 335)
        paths = (_disjoint_paths(0, 1),)
        paths_by_rank = tuple((path,) for path in paths[0])
        stream_only = _shard_stream_ns_shards(
            rates, LinkFlowRegistry(), None,
            paths_by_rank=paths_by_rank,
            bytes_by_rank=(360_000 - 25_600, 40_000 - 25_600),
            include_self=True, kind="read")
        with_latency = _transfer_ns_shards(
            rates, LinkFlowRegistry(), None,
            paths_by_rank=paths_by_rank,
            bytes_by_rank=(25_600, 25_600),
            include_self=True, kind="read")
        self.assertEqual(stream_only, 335)
        self.assertEqual(with_latency, 26 + 10)


# ============================================ 6. 代表路径广播锚（C4 桥接）==


class RepresentativeBroadcastParityTest(unittest.TestCase):
    """route_paths_fn 未注入（单测/离线代表路径）：字节均衡、空链路时
    逐 shard 广播口径与旧聚合口径同 wall 值——C4 金值重推导的桥接锚。"""

    def test_balanced_bytes_match_legacy_aggregate(self):
        # 代表路径广播 → 除数 = 2（自身重叠按 rank 数计）→ 每 shard
        # 3000/(10/2) = 600 → wall = 10 + 600 ≡ 旧聚合
        # _transfer_ns(6000, hops=1, divisor=1) = 10 + 600。
        model = _model(rates=_rates(noc=10.0, hbm=1e5), route_paths_fn=None)
        candidate = model.estimate_P_with_continuation(
            session=_session(history_bytes=(3000, 3000)),
            request=_request(), instance_index=1,
            action=ACTION_COPY, remote_enabled=True)
        legacy = _transfer_ns(
            total_bytes=6000, path_hops=1, divisor=1,
            rates=model.rates, per_hop_latency_ns=None, startup_ns=0)
        self.assertEqual(candidate.breakdown.history_prep_ns, legacy)
        self.assertEqual(candidate.breakdown.history_prep_ns, 610)

    def test_pool_suffix_leg_per_shard(self):
        # D04（2026-09-26）：池后缀逐 shard 计价——每 rank 1000 B、退
        # 化口径独立池端口（除数 1）→ 100 + max(1000/5, 1000/1e5) =
        # 300（旧聚合口径 pool(2000)=500 随 _pool_transfer_ns 废除）。
        model = _model(rates=_rates(noc=10.0, hbm=1e5), route_paths_fn=None)
        candidate = model.estimate_P_with_continuation(
            session=_session(history_bytes=(3000, 3000),
                             missing=(1000, 1000),
                             location="partial_hbm_remote", prefix=3),
            request=_request(), instance_index=1,
            action=ACTION_COPY, remote_enabled=True)
        expected_pool = _pool_transfer_ns_shards(
            model.rates, model._noc_registry, model.hbm_port_registry,
            kind="pool_restore",
            edge_ranks_by_rank=(0, 1),
            paths_by_rank=((0,), (1,)),
            bytes_by_rank=(1000, 1000))
        self.assertEqual(
            candidate.breakdown.history_prep_ns, 610 + expected_pool)
        self.assertEqual(expected_pool, 100 + 200)


if __name__ == "__main__":
    unittest.main()
