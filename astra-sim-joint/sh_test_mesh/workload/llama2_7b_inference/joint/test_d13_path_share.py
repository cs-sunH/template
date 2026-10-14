#!/usr/bin/env python3
"""test_joint_d13_path_share.py -- D13（v2.0 C06/§9.2，2026-09-27）：
每路径资源份额、去掉吞吐倒推流数——零后端单测。

覆盖（条目验收三点 + C06 目标）：
  1. **单路径拥塞不错误减速无关路径**：``CandidatePathNocShare`` 逐
     shard 路径求值瓶颈份额——只经过热点链路的 shard 被减速，不经过
     热点的 shard 保持独占带宽（旧"全部路径并集取一个瓶颈除数统一使
     用"的过度聚合废除）；
  2. **旧流与候选不重计**：在册旧流 + 候选自身共链份额按并发数叠加
     （注册侧不含候选、候选由 self_counts 承载），遥测流数与在册流
     max 去重（同一流量不重复计数）；
  3. **缺流数不由"总吞吐满载"推断单流**：速率-only 链路 = 显式未知
     ——不产生除数腿、退在册流近似（unknown_flow_count_links 披露）；
  4. **复用在线逐 shard 路由身份**（C11）：``register_path``（与
     sh30 ``_register_transfer_flows`` 同一登记通道，逐 shard 完整
     noc_path）登记的流按链路键被每路径视图读到——只影响经过该链路
     的 shard 路径；
  5. **查询共用**：同路径/同链路查询共享 memo 结果（一次求值）。

运行：cd sh_test_mesh/workload/llama2_7b_inference &&
      python3 joint/test_joint_d13_path_share.py   （或 pytest 同路径）
"""
import os
import sys
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
for _p in (_HERE, _PARENT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from joint.joint_cost_model import (  # noqa: E402
    CandidatePathNocShare,
    JointHardwareRates,
    LinkFlowRegistry,
    _transfer_ns_shards,
)


def _rates(noc=10.0, hbm=1e9, lat=0):
    return JointHardwareRates.from_gbps(
        noc_link_gbps=noc, pool_port_gbps=5.0, local_hbm_gbps=hbm,
        d2d_latency_ns=lat, pool_latency_ns=100)


# 两条不相交单跳 shard 路径（rank 0/1 各走自己的链路）。
PATH_A = (0, 2)     # rank 0 的路径
PATH_B = (1, 3)     # rank 1 的路径——与 A 无共享链路


class PerPathDecouplingTest(unittest.TestCase):
    """验收①：只有一条路径拥塞时，不相关路径不被错误减速。"""

    def test_congested_path_does_not_slow_unrelated_shard(self):
        # 背景流全在 A 路径（链路 (0,2) 注册 3 条旧流）；字节向量偏
        # 向 B（B 独走时更慢）。每路径份额：A 除数 = 3+自身 1 = 4、
        # B 除数 = 1 → wall = max(100·4/10, 200·1/10) = 40；
        # 旧并集除数把热点外溢给 B：两 shard 都除 4 → wall = 80。
        registry = LinkFlowRegistry()
        for _ in range(3):
            registry.register(0, 2)
        share = CandidatePathNocShare(
            registry, (PATH_A, PATH_B), include_self=True)
        self.assertEqual(share.path_divisor(PATH_A), 4.0)
        self.assertEqual(share.path_divisor(PATH_B), 1.0)
        self.assertEqual(share.max_path_divisor(), 4.0)
        wall = _transfer_ns_shards(
            _rates(), registry, None,
            paths_by_rank=((PATH_A,), (PATH_B,)),
            bytes_by_rank=(100, 200),
            include_self=True, kind="copy")
        self.assertEqual(wall, 40)
        self.assertLess(wall, 80)   # 旧并集口径的减速值（不再发生）

    def test_per_path_view_used_by_transfer_primitive(self):
        # 原语级：内部构造（不传视图）同享每路径份额——热点链路只减
        # 速经过它的 shard。
        registry = LinkFlowRegistry()
        for _ in range(2):
            registry.register(0, 2)
        wall = _transfer_ns_shards(
            _rates(), registry, None,
            paths_by_rank=((PATH_A,), (PATH_B,)),
            bytes_by_rank=(100, 300),
            include_self=True, kind="copy")
        # A = 100·(2+1)/10 = 30；B = 300·1/10 = 30 → wall = 30
        # （旧并集除数 3 作用到 B：300·3/10 = 90）。
        self.assertEqual(wall, 30)
        self.assertLess(wall, 90)

    def test_shared_link_self_overlap_billing_preserved(self):
        # 共链候选流互相计费不因每路径化丢失：6 条 TP 流同走一条链路
        # → 每 shard 除数 6（自身并发 6），wall = 6× 单 shard。
        registry = LinkFlowRegistry()
        paths = (((0, 6),),) * 6
        wall = _transfer_ns_shards(
            _rates(), registry, None,
            paths_by_rank=paths, bytes_by_rank=(60,) * 6,
            include_self=True, kind="copy")
        self.assertEqual(wall, 36)
        # include_self=False：候选份额剥离 → 除数回落 1。
        wall_no_self = _transfer_ns_shards(
            _rates(), registry, None,
            paths_by_rank=paths, bytes_by_rank=(60,) * 6,
            include_self=False, kind="copy")
        self.assertEqual(wall_no_self, 6)

    def test_zero_byte_shard_not_in_share_family(self):
        # 零字节 shard 无流、不进自身份额族（既有过滤纪律逐路径保持）。
        registry = LinkFlowRegistry()
        share = CandidatePathNocShare(
            registry, (PATH_A,), include_self=True)
        self.assertEqual(share.path_divisor(PATH_A), 1.0)
        wall = _transfer_ns_shards(
            _rates(), registry, None,
            paths_by_rank=((PATH_A,), (PATH_A,)),
            bytes_by_rank=(6000, 0),
            include_self=True, kind="copy")
        self.assertEqual(wall, 600)


class TelemetryPerPathTest(unittest.TestCase):
    """流数口径遥测 × 每路径视图：覆盖哪条链路就抬哪条路径。"""

    def test_flow_count_lifts_only_covered_path(self):
        registry = LinkFlowRegistry()
        view = registry.with_effective_rates(
            {}, link_capacity_bytes_per_ns=10.0,
            link_flow_counts={PATH_A: 5.0})
        share = CandidatePathNocShare(view, (PATH_A, PATH_B),
                                      include_self=True)
        # 遥测流数只抬 A：max(在册 0, 流数 5) + 自身 1 = 6；B 不受染。
        self.assertEqual(share.path_divisor(PATH_A), 6.0)
        self.assertEqual(share.path_divisor(PATH_B), 1.0)
        wall = _transfer_ns_shards(
            _rates(), view, None,
            paths_by_rank=((PATH_A,), (PATH_B,)),
            bytes_by_rank=(100, 200),
            include_self=True, kind="copy", noc_share=share)
        # A = 100·6/10 = 60；B = 200/10 = 20 → wall = 60。
        self.assertEqual(wall, 60)

    def test_telemetry_and_registered_max_dedup(self):
        # 验收②：旧流与遥测/候选不重计——在册 4 流 + 遥测流数 1.2 →
        # max 取 4（遥测多计不放大）；+ 候选自身 1 → 5。
        registry = LinkFlowRegistry()
        for _ in range(4):
            registry.register(*PATH_A)
        view = registry.with_effective_rates(
            {}, link_capacity_bytes_per_ns=10.0,
            link_flow_counts={PATH_A: 1.2})
        share = CandidatePathNocShare(view, (PATH_A, PATH_B),
                                      include_self=True)
        self.assertEqual(share.path_divisor(PATH_A), 5.0)


class RateOnlyUnknownTest(unittest.TestCase):
    """验收③：缺流数不由"总吞吐满载"推断单流——显式未知。"""

    def test_rate_only_link_reports_unknown(self):
        # 速率-only（capacity/rate 名义 100）不产生除数腿：视图份额 =
        # 在册流近似。
        registry = LinkFlowRegistry()
        registry.register(*PATH_A)   # 背景 1 流
        view = registry.with_effective_rates(
            {PATH_A: 0.1}, link_capacity_bytes_per_ns=10.0)
        self.assertIsNone(view.effective_divisor(PATH_A))
        share = CandidatePathNocShare(view, (PATH_A, PATH_B),
                                      include_self=True)
        # 名义 100 若冒充流数会把 A 抬到 101；在册流近似 → 1+1 = 2。
        self.assertEqual(share.path_divisor(PATH_A), 2.0)
        self.assertEqual(share.path_divisor(PATH_B), 1.0)
        self.assertEqual(view.disclosure()["unknown_flow_count_links"], 1)

    def test_rate_only_no_lift_with_empty_registry(self):
        # 空注册表 + 速率-only：视图份额下界 1（独占带宽，非名义换算）。
        registry = LinkFlowRegistry()
        view = registry.with_effective_rates(
            {PATH_A: 5.0}, link_capacity_bytes_per_ns=10.0)
        share = CandidatePathNocShare(view, (PATH_A,), include_self=True)
        self.assertEqual(share.path_divisor(PATH_A), 1.0)
        self.assertIsNone(view.effective_divisor(PATH_A))


class RouteIdentityReuseTest(unittest.TestCase):
    """C11：``register_path``（与 sh30 ``_register_transfer_flows`` 同
    一逐 shard 完整 noc_path 登记通道）的流被每路径视图按链路键读到。"""

    def test_register_path_flows_feed_per_path_shares(self):
        registry = LinkFlowRegistry()
        # 两次 #readplan 型登记走 A 路径、一次走 B 路径（逐 shard 完整
        # 路由身份 → 链路键 (0,2)/(1,3)）。
        registry.register_path(PATH_A, owner="r1#readplan")
        registry.register_path(PATH_A, owner="r1#readplan")
        registry.register_path(PATH_B, owner="r2#readplan")
        share = CandidatePathNocShare(registry, (PATH_A, PATH_B),
                                      include_self=True)
        self.assertEqual(share.path_divisor(PATH_A), 3.0)   # 2 旧 + 自身
        self.assertEqual(share.path_divisor(PATH_B), 2.0)   # 1 旧 + 自身

    def test_registry_state_untouched_by_share_view(self):
        # 决策时点只读快照纪律：视图构造/查询零注册表状态变更。
        registry = LinkFlowRegistry()
        registry.register_path((0, 1, 2), owner="r1")
        before = registry.snapshot()
        share = CandidatePathNocShare(registry, ((0, 1, 2),),
                                      include_self=True)
        self.assertGreaterEqual(share.path_divisor((0, 1, 2)), 2.0)
        self.assertEqual(registry.snapshot(), before)


class QuerySharingTest(unittest.TestCase):
    """相同路径/阶段的查询共用结果（C06：视图 memo）。"""

    def test_link_base_lookup_shared_across_queries(self):
        registry = _CountingBaseFlows()
        registry.register(0, 1)
        registry.register(1, 2)
        share = CandidatePathNocShare(registry, ((0, 1, 2), (0, 1, 2)),
                                      include_self=True)
        first = share.path_divisor((0, 1, 2))
        second = share.path_divisor((0, 1, 2))
        self.assertEqual(first, second)
        # 两跳链路各查一次在册基；第二次路径查询命中缓存不再穿透。
        self.assertEqual(registry.base_calls, 2)

    def test_disclosure_shape(self):
        registry = LinkFlowRegistry()
        share = CandidatePathNocShare(registry, (PATH_A, PATH_B),
                                      include_self=True)
        report = share.disclosure()
        self.assertEqual(report["paths"], 2)
        self.assertTrue(report["include_self"])
        self.assertEqual(report["family_links"], 2)
        self.assertEqual(report["max_path_divisor"], 1.0)


class _CountingBaseFlows(LinkFlowRegistry):
    """link_old_flows 调用计数探针（视图 memo 断言用）。"""

    def __init__(self):
        super().__init__()
        self.base_calls = 0

    def link_old_flows(self, link):
        self.base_calls += 1
        return super().link_old_flows(link)


if __name__ == "__main__":
    unittest.main()
