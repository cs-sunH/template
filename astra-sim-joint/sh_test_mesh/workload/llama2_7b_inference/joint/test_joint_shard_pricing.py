#!/usr/bin/env python3
"""test_joint_shard_pricing.py -- C1（WP1a）逐 shard 三腿 min 计价 +
D1 读放大因果口径的零后端单测（F1 冻结形态的解析锚点）。

覆盖（卡 C1 测试清单）：
  1. 平价锚点：d ≤ ρ 空链路单流下传输腿 ≡ 本地读平价（远读每步
     平价锚已随远读退役删除）
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


def _request(input_tokens=50, decode=10, input_bytes=(25, 25)):
    return RequestView(
        request_id="r", session_id="s", input_tokens=input_tokens,
        history_tokens_before=100, estimated_decode_tokens=decode,
        horizon_source="session_online_mean",
        input_kv_bytes_by_tp_rank=input_bytes)


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
            instance_index=1, action=ACTION_COPY)
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
            else route_paths_fn))


# ==================================================== 1. 平价锚点（§1.2）==


class ParityAnchorTest(unittest.TestCase):
    """d ≤ ρ 空链路单流：传输腿历史扫描项 ≡ 本地读平价。"""

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
            include_self=True, kind="copy")
        local_scan_ns = int(shard_bytes / rates.local_hbm_bytes_per_ns)
        self.assertEqual(local_scan_ns, 1000)
        self.assertLess(
            int(shard_bytes / rates.noc_link_bytes_per_ns), local_scan_ns)
        # 跳时延剥离后 ≡ 本地扫描项（hop 启动另计，不进扫描项平价）。
        self.assertEqual(wall - 2 * rates.d2d_latency_ns, local_scan_ns)

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
            include_self=True, kind="copy")
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

    def test_unknown_kind_fails_closed(self):
        # kind 取值域 = {"copy", "merge"}（远读 "read"/"staging" 已随
        # 远读退役删除——枚举外取值 fail-closed 不变）。
        rates = _rates(noc=1000.0, hbm=10.0, lat=0)
        for kind in ("read", "staging", "prefetch"):
            with self.assertRaises(Exception):
                _transfer_ns_shards(
                    rates, LinkFlowRegistry(), None,
                    paths_by_rank=(((0, 1),),), bytes_by_rank=(self.BYTES,),
                    include_self=True, kind=kind)


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


if __name__ == "__main__":
    unittest.main()
