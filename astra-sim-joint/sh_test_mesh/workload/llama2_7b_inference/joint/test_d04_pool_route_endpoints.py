#!/usr/bin/env python3
"""test_d04_pool_route_endpoints.py -- D04 池路径逐 shard 计价零后端单测
（2026-09-26，v2.0 C02/C11/式 7：池路径补逐 rank 路由与 HBM 端点）。

覆盖（对应 joint修改文档.md D04 验收三条 + 修改位置钉测）：
  1. 远/近实例恢复代价有区别：恢复 wall 携带真实 edge→目标 NoC 路径
     （λ = 池时延 + 逐跳时延；带宽项 = 池端口 ∪ NoC ∪ 目标 HBM 取
     max，式 7）——远实例恢复更慢；
  2. 共享/独立池端口有区别：TP 并行 shard 共享同一边缘端口时按全部
     自身流计（divisor = 自身 shard 数 + 在册他流），独立端口各得独
     占份额——共享端口 wall 严格更大；
  3. 恢复写+本地消费不超端口服务能力：remote_load 目标 HBM 端点入
     HbmPortFlowRegistry（sh30 _register_transfer_flows 登记边界，
     D04 起 pool 路径同表）后，候选恢复写腿除数 = 活跃 decode + 在
     册他流 + 自身 1——u+1 条流各得 B/(u+1)，合计恰不超端口峰值；
  4. 式 (7) 逐项钉：目标 HBM 慢时恢复反映目标写瓶颈（HBM 腿入
     max）、写回方向 = 源 HBM 读 + 到边界 NoC + 池端口、零字节无事
     务、退化口径（未注入拓扑 fn：池端口即在 rank 本地，无 NoC 腿）；
  5. 修复轮三（2026-09-27）：sh30 执行侧（_register_transfer_flows）
     零距离池事务端点登记——remote_load/remote_store 单 rank noc_path
     仍登记端点 HBM（divisor==1、release 归零；与计价侧无条件收端
     点腿同口径），noc_migrate 单 rank 仍零登记（门内保留），多跳
     行为不变。

手算锚硬件：B_link = 10 B/ns、B_pool = 5 B/ns、B_HBM = 100 B/ns
（慢 HBM 夹具 2/1 B/ns）、单跳时延 10 ns、池端口事务时延 100 ns。

运行：cd sh_test_mesh/workload/llama2_7b_inference &&
      python3 joint/test_d04_pool_route_endpoints.py   （或 pytest）
"""
import os
import sys
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
for _p in (_HERE, _PARENT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from joint.hbm_port_flow_registry import HbmPortFlowRegistry
from joint.joint_cost_model import (  # noqa: E402
    InstanceLoadView,
    JointCostError,
    JointCostModel,
    JointHardwareRates,
    LinkFlowRegistry,
    ServiceFactors,
    SessionKVView,
    RequestView,
    _pool_edge_divisor,
    _pool_transfer_ns_shards,
)


# ============================================================ 夹具构造 ==

def _rates(pool_latency_ns=100, d2d_latency_ns=10,
           noc=10.0, pool=5.0, hbm=100.0):
    return JointHardwareRates.from_gbps(
        noc_link_gbps=noc, pool_port_gbps=pool, local_hbm_gbps=hbm,
        d2d_latency_ns=d2d_latency_ns, pool_latency_ns=pool_latency_ns)


# rank 空间：池边缘端口 0/5；实例 0 = 近（rank 1 一跳到 edge 0、
# rank 2 一跳到 edge 5——独立端口）；实例 1 = 远（rank 3 两跳到
# edge 0、rank 4 一跳到 edge 5）。
NEAREST_EDGE = {0: 0, 1: 0, 2: 5, 3: 0, 4: 5, 5: 5}
ROUTES = {
    (0, 1): (0, 1), (5, 2): (5, 2), (0, 3): (0, 1, 3), (5, 4): (5, 4),
    (1, 0): (1, 0), (2, 5): (2, 5), (3, 0): (3, 1, 0), (4, 5): (4, 5),
}
INSTANCE_RANKS = {0: (1, 2), 1: (3, 4)}


def _rank_route(source, target):
    return ROUTES[(source, target)]


def _model(loads, *, rates=None, pool_edge_flows=None,
           hbm_port_registry=None, nearest_edge=None,
           rank_route=None, instance_ranks=None):
    flows = {0: 0, 5: 0}
    flows.update(pool_edge_flows or {})

    def pool_edge_flows_fn(edge):
        return flows.get(edge, 0)

    return JointCostModel(
        rates=rates if rates is not None else _rates(),
        loads=loads, flow_registry=LinkFlowRegistry(),
        service_factors=ServiceFactors(),
        prefill_ns_per_token=1.0, decode_ns_per_token=2.0,
        model_layers=4, instance_tp_size=2,
        route_fn=lambda source, target: ((source, target), 1),
        nearest_edge_fn=(
            nearest_edge if nearest_edge is not None
            else (lambda rank: NEAREST_EDGE[rank])),
        rank_route_fn=(
            rank_route if rank_route is not None else _rank_route),
        instance_ranks_fn=(
            instance_ranks if instance_ranks is not None
            else (lambda index: INSTANCE_RANKS[index])),
        pool_edge_flows_fn=pool_edge_flows_fn,
        hbm_port_registry=hbm_port_registry)


def _load(index):
    return InstanceLoadView(
        instance_index=index, queued_task_load_ns=0,
        running_task_load_ns=0, active_decode_task_load_ns=0,
        hbm_remaining_bytes_by_tp_rank=(10 ** 9, 10 ** 9),
        reclaimable_bytes_by_tp_rank=(10 ** 9, 10 ** 9))


def _session(location="remote_memory", history_bytes=(0, 0),
             missing=(1000, 1000), resident=None, prefix=0):
    return SessionKVView(
        session_id="s", home_instance=1, resident_instance=resident,
        location=location, history_tokens=100, resident_prefix_layers=prefix,
        history_bytes_by_tp_rank=history_bytes,
        missing_bytes_by_tp_rank=missing)


def _request():
    return RequestView(
        request_id="r", session_id="s", input_tokens=5,
        history_tokens_before=100, estimated_decode_tokens=0,
        horizon_source="run_online_mean",
        input_kv_bytes_by_tp_rank=(25, 25))


# ==================================================== 验收 1：远/近有区别 ==


class FarNearRestoreCostTest(unittest.TestCase):
    """相同恢复量落到远/近实例的路径代价有区别（NoC 腿逐跳时延 +
    路径资源集进式 (7) max——旧聚合口径无路径几何、远近同价）。"""

    def test_far_instance_restore_costs_more_by_hops(self):
        model = _model({0: _load(0), 1: _load(1)})
        near = model.estimate_P_with_continuation(
            session=_session(), request=_request(), instance_index=0,
            action="copy", remote_enabled=True)
        far = model.estimate_P_with_continuation(
            session=_session(), request=_request(), instance_index=1,
            action="copy", remote_enabled=True)
        self.assertTrue(near.applicable and far.applicable)
        # 近实例：两 shard 各 1 跳 → wall = (100 + 10) + max(200, 100,
        # 10) = 310（池端口项主导；除数 1 独立端口）。
        self.assertEqual(near.breakdown.history_prep_ns, 110 + 200)
        # 远实例：rank 3 shard 两跳（λ = 120）→ wall = 120 + 200 = 320。
        self.assertEqual(far.breakdown.history_prep_ns, 120 + 200)
        self.assertGreater(
            far.breakdown.history_prep_ns,
            near.breakdown.history_prep_ns)
        # 差值逐位 = 最远 shard 的逐跳时延增量（1 hop × 10 ns）。
        self.assertEqual(
            far.breakdown.history_prep_ns
            - near.breakdown.history_prep_ns, 10)

    def test_restore_noc_path_resources_enter_lambda(self):
        # 直接原语钉：路径 (0,1,3) 的 2 跳时延计入 λ（式 7 λ_f 含逐
        # 跳时延）；NoC 项（1000/10 = 100）与 HBM 项（10）均不越过池
        # 端口项（200）——wall = 100 + 20 + 200 = 320。
        wall = _pool_transfer_ns_shards(
            _rates(), LinkFlowRegistry(), None,
            kind="pool_restore",
            edge_ranks_by_rank=(0, 5),
            paths_by_rank=((0, 1, 3), (5, 4)),
            bytes_by_rank=(1000, 1000))
        self.assertEqual(wall, 100 + 20 + 200)


# ==================================================== 验收 2：池端口份额 ==


class SharedVsIndependentPoolPortTest(unittest.TestCase):
    """两个 rank 共享一个池端口与使用两个独立端口不同（自身 shard 经
    同一端口按全部自身流计；divisor = 自身数 + 在册他流）。"""

    def test_shared_edge_port_halves_per_shard_share(self):
        # 共享 edge 0（两 shard 路径 (0,1)/(0,2) 各 1 跳）：自身
        # 2 shard → 除数 2 → 每 shard 池项 = 1000/(5/2) = 400 →
        # wall = 110 + 400 = 510。
        shared = _model(
            {0: _load(0)},
            nearest_edge=lambda rank: 0,
            rank_route=lambda a, b: {
                (0, 1): (0, 1), (0, 2): (0, 2)}[(a, b)],
            instance_ranks=lambda index: (1, 2))
        wall_shared = shared._pool_transfer_ns_by_rank(
            0, (1000, 1000), restore=True)
        self.assertEqual(wall_shared, 110 + 400)

    def test_independent_edge_ports_full_share_each(self):
        # 独立端口（edge 0 / edge 5，各 1 跳）：除数各 1 → 池项 200 →
        # wall = 110 + 200 = 310——严格小于共享端口形态 510。
        independent = _model(
            {0: _load(0)},
            nearest_edge=lambda rank: {1: 0, 2: 5}[rank],
            rank_route=lambda a, b: {
                (0, 1): (0, 1), (5, 2): (5, 2)}[(a, b)],
            instance_ranks=lambda index: (1, 2))
        wall_independent = independent._pool_transfer_ns_by_rank(
            0, (1000, 1000), restore=True)
        self.assertEqual(wall_independent, 110 + 200)
        self.assertLess(110 + 200, 110 + 400)

    def test_registered_inflight_flows_lift_edge_divisor(self):
        # 在册他流（他会话在途池流）按流查询实际 edge 计入：共享端口
        # 除数 = 自身 2 + 在册 3 = 5 → 每 shard 池项 = 1000/(5/5) =
        # 1000 → wall = 110 + 1000。
        contended = _model(
            {0: _load(0)}, pool_edge_flows={0: 3},
            nearest_edge=lambda rank: 0,
            rank_route=lambda a, b: {
                (0, 1): (0, 1), (0, 2): (0, 2)}[(a, b)],
            instance_ranks=lambda index: (1, 2))
        self.assertEqual(
            contended._pool_transfer_ns_by_rank(0, (1000, 1000),
                                                restore=True),
            110 + 1000)

    def test_pool_edge_divisor_primitive(self):
        # 原语逐位：own 按发流 shard 计（零字节 shard 不占端口）、
        # others 为注入在册值、下界 1。
        self.assertEqual(
            _pool_edge_divisor(0, (0, 0), (1000, 1000), None), 2)
        self.assertEqual(
            _pool_edge_divisor(0, (0, 5), (1000, 1000), lambda e: 3), 4)
        self.assertEqual(
            _pool_edge_divisor(7, (0, 5), (1000, 0), None), 1)


# ============================================ 验收 3：恢复写+本地消费 ≤ 端口 ==


class RestoreWriteWithinPortCapacityTest(unittest.TestCase):
    """remote_load 真实 HBM 端点登记后，候选恢复写与本地消费（活跃
    decode + 在册他流）共用目标端口：候选份额 = B/(u+1)，u+1 条流合
    计恰不超端口峰值——候选不再独占端口。"""

    def _registry(self, active_streams):
        registry = HbmPortFlowRegistry()
        registry.attach_active_decode_provider(lambda port: active_streams)
        return registry

    def test_candidate_restore_shares_port_with_local_consumption(self):
        # 目标 rank 1 端口：活跃 decode 1 流 + 在册他流恢复写 1 条 →
        # u(1) = 2；慢 HBM（2 B/ns）令写腿主导：rank 1 shard 写腿 =
        # 1000/(2/3) = 1500 → wall = 110 + 1500（rank 2 端口 u = 1 →
        # 写腿 1000 不触上界）。3 条并发流各 2/3 B/ns、合计 2 B/ns =
        # 端口峰值——恢复写+本地消费不超服务能力（等份饱和）。
        registry = self._registry(active_streams=1)
        registry.register(1, "other#restore")
        model = _model({0: _load(0)}, rates=_rates(hbm=2.0),
                       hbm_port_registry=registry)
        wall = model._pool_transfer_ns_by_rank(0, (1000, 1000),
                                               restore=True)
        self.assertEqual(wall, 110 + 1500)
        candidate_rate = 1000 / 1500
        consumers = int(registry.divisor(1)) + 1
        self.assertEqual(consumers, 3)
        self.assertEqual(consumers * candidate_rate, 2.0)

    def test_uncontended_restore_gets_full_write_leg(self):
        # 对照锚：无消费（u = 0）→ 写腿独占 = 1000/2 = 500 → wall =
        # 110 + 500 < 争用形态——端口占用真实抬升候选代价。
        registry = self._registry(active_streams=0)
        model = _model({0: _load(0)}, rates=_rates(hbm=2.0),
                       hbm_port_registry=registry)
        self.assertEqual(
            model._pool_transfer_ns_by_rank(0, (1000, 1000), restore=True),
            110 + 500)

    def test_sh30_remote_load_registration_feeds_pricing_divisor(self):
        # sh30 接线：remote_load 经 _register_transfer_flows 登记目标
        # HBM 写端点（D04）→ 注册表除数抬升；完成事件
        # _release_transfer_flows 注销 → 除数回落 0（计价侧经
        # hbm_port_registry 注入同一对象——端点争用因果可见）。
        from online.sh30_online_scheduler import Sh30OnlineScheduler
        scheduler = Sh30OnlineScheduler.__new__(Sh30OnlineScheduler)
        scheduler._joint_flows = LinkFlowRegistry()

        class _PoolPorts:
            def __init__(self):
                self.counts = {}

            def register(self, edge, *, owner):
                self.counts[edge] = self.counts.get(edge, 0) + 1

            def release_owner(self, owner):
                return 0

        scheduler._pool_ports = _PoolPorts()
        scheduler._hbm_ports = HbmPortFlowRegistry()
        from face_scheduler import KVTransfer, KVTransferShard
        transfer = KVTransfer(
            kind="remote_load", phase="history", reason="test",
            session_id="s", trigger_request_id="r",
            source_instance_index=None, target_instance_index=0,
            total_bytes=1000,
            shards=(KVTransferShard(
                source_rank=0, target_rank=1, edge_rank=0, bytes=1000,
                noc_path=(0, 1), layer_start=0, layer_end=2),),
            model_layers=2, layer_start=0, layer_end=2,
            resident_prefix_layers_before=0, resident_prefix_layers_after=2)
        self.assertEqual(scheduler._hbm_ports.divisor(1), 0)
        scheduler._register_transfer_flows((transfer,), owner="r#restore")
        self.assertEqual(scheduler._hbm_ports.divisor(1), 1)
        scheduler._release_transfer_flows("r#restore")
        self.assertEqual(scheduler._hbm_ports.divisor(1), 0)


# =========================== 修复轮三：执行侧零距离端点登记（sh30 侧）==


class ZeroDistanceEndpointRegistrationTest(unittest.TestCase):
    """sh30 执行侧（_register_transfer_flows）零距离池事务的端点 HBM
    登记（修复轮三，2026-09-27）：remote_load/remote_store 端点登记
    **无条件**（不再要求 len(noc_path) >= 2）——零距离池事务（edge ==
    端点 rank、deterministic_xy_route 同端点返回单 rank 路径）生产可
    达，旧门漏登端点 HBM 与计价侧无条件收端点腿（
    _pool_transfer_ns_shards）失衡（候选 u_port 低估）；noc_migrate
    仍限多跳门内（零距离无生产构造点；HbmPortFlowRegistry.register
    纯累加无去重，移出门会对同一 rank 双登记抬升除数）。"""

    def _scheduler(self):
        # 接线替身同 test_sh30_remote_load_registration_feeds_pricing_
        # divisor 手法：__new__ 直构 + 三件套（流/池端口/HBM 端口注
        # 册表）。
        from online.sh30_online_scheduler import Sh30OnlineScheduler
        scheduler = Sh30OnlineScheduler.__new__(Sh30OnlineScheduler)
        scheduler._joint_flows = LinkFlowRegistry()

        class _PoolPorts:
            def __init__(self):
                self.counts = {}

            def register(self, edge, *, owner):
                self.counts[edge] = self.counts.get(edge, 0) + 1

            def release_owner(self, owner):
                return 0

        scheduler._pool_ports = _PoolPorts()
        scheduler._hbm_ports = HbmPortFlowRegistry()
        return scheduler

    @staticmethod
    def _transfer(kind, shard):
        from face_scheduler import KVTransfer
        return KVTransfer(
            kind=kind, phase="history", reason="test",
            session_id="s", trigger_request_id="r",
            source_instance_index=None, target_instance_index=0,
            total_bytes=shard.bytes, shards=(shard,),
            model_layers=2, layer_start=0, layer_end=2,
            resident_prefix_layers_before=0,
            resident_prefix_layers_after=2)

    def test_zero_distance_remote_load_registers_target_endpoint(self):
        # 零距离 remote_load（edge == 目标 rank；生产形态 source=edge、
        # noc_path = route(edge, edge) 单 rank）：目标 HBM 写端点仍登
        # 记 → divisor(target) == 1；release_owner 后归零（同 owner
        # 同生命周期）。池端口（edge）照常占用，登记/注销只涉及端点
        # HBM 侧归零。
        from face_scheduler import KVTransferShard
        scheduler = self._scheduler()
        transfer = self._transfer("remote_load", KVTransferShard(
            source_rank=1, target_rank=1, edge_rank=1, bytes=1000,
            noc_path=(1,), layer_start=0, layer_end=2))
        self.assertEqual(scheduler._hbm_ports.divisor(1), 0)
        scheduler._register_transfer_flows((transfer,), owner="r#restore")
        self.assertEqual(scheduler._hbm_ports.divisor(1), 1)
        self.assertEqual(scheduler._pool_ports.counts.get(1), 1)
        scheduler._release_transfer_flows("r#restore")
        self.assertEqual(scheduler._hbm_ports.divisor(1), 0)

    def test_zero_distance_remote_store_registers_source_endpoint(self):
        # 零距离 remote_store（source == edge、noc_path = (source,)）：
        # 源 HBM 读端点仍登记 → divisor(source) == 1；release 归零。
        from face_scheduler import KVTransferShard
        scheduler = self._scheduler()
        transfer = self._transfer("remote_store", KVTransferShard(
            source_rank=5, target_rank=5, edge_rank=5, bytes=1000,
            noc_path=(5,), layer_start=0, layer_end=2))
        self.assertEqual(scheduler._hbm_ports.divisor(5), 0)
        scheduler._register_transfer_flows((transfer,), owner="r#evict")
        self.assertEqual(scheduler._hbm_ports.divisor(5), 1)
        scheduler._release_transfer_flows("r#evict")
        self.assertEqual(scheduler._hbm_ports.divisor(5), 0)

    def test_zero_distance_noc_migrate_stays_gated(self):
        # noc_migrate 分支留在 len >= 2 门内：单 rank noc_path 的
        # noc_migrate 零登记（零距离 noc_migrate 无生产构造点；注册
        # 表 register 纯累加无去重，移出门会对同一 rank 双登记抬升
        # 除数——本钉测锁死该门）。
        from face_scheduler import KVTransferShard
        scheduler = self._scheduler()
        transfer = self._transfer("noc_migrate", KVTransferShard(
            source_rank=1, target_rank=1, edge_rank=None, bytes=1000,
            noc_path=(1,), layer_start=0, layer_end=2))
        scheduler._register_transfer_flows((transfer,), owner="r#migrate")
        self.assertEqual(scheduler._hbm_ports.divisor(1), 0)
        scheduler._release_transfer_flows("r#migrate")
        self.assertEqual(scheduler._hbm_ports.divisor(1), 0)

    def test_multihop_registration_behavior_unchanged(self):
        # 对照（多跳行为不变）：remote_load 多跳只登末端目标端点、
        # remote_store 多跳只登首端源端点、noc_migrate 多跳双端点；
        # release 全体归零。
        from face_scheduler import KVTransferShard
        scheduler = self._scheduler()
        load = self._transfer("remote_load", KVTransferShard(
            source_rank=0, target_rank=3, edge_rank=0, bytes=1000,
            noc_path=(0, 1, 3), layer_start=0, layer_end=2))
        store = self._transfer("remote_store", KVTransferShard(
            source_rank=4, target_rank=5, edge_rank=5, bytes=1000,
            noc_path=(4, 5), layer_start=0, layer_end=2))
        migrate = self._transfer("noc_migrate", KVTransferShard(
            source_rank=1, target_rank=2, edge_rank=None, bytes=1000,
            noc_path=(1, 2), layer_start=0, layer_end=2))
        scheduler._register_transfer_flows(
            (load, store, migrate), owner="r#multi")
        self.assertEqual(scheduler._hbm_ports.divisor(3), 1)  # load 末端
        self.assertEqual(scheduler._hbm_ports.divisor(0), 0)
        self.assertEqual(scheduler._hbm_ports.divisor(4), 1)  # store 首端
        self.assertEqual(scheduler._hbm_ports.divisor(5), 0)
        self.assertEqual(scheduler._hbm_ports.divisor(1), 1)  # migrate 双端
        self.assertEqual(scheduler._hbm_ports.divisor(2), 1)
        scheduler._release_transfer_flows("r#multi")
        for rank in (0, 1, 2, 3, 4, 5):
            self.assertEqual(scheduler._hbm_ports.divisor(rank), 0)


# ======================================================= 式 (7) 逐项钉 ==


class PoolTransferFormulaPinsTest(unittest.TestCase):
    """式 (7) X_f = λ_f + max_e D_f,e/μ_e 的资源集方向契约。"""

    def test_slow_target_hbm_write_dominates_restore(self):
        # 目标 HBM 慢：写腿 1000/1 = 1000 主导池项 200/NoC 项 100 →
        # 恢复 wall 反映目标写瓶颈（restore 资源集含目标 HBM）。
        wall = _pool_transfer_ns_shards(
            _rates(hbm=1.0), LinkFlowRegistry(), None,
            kind="pool_restore",
            edge_ranks_by_rank=(0, 0),
            paths_by_rank=((0, 1), (0, 2)),
            bytes_by_rank=(1000, 1000))
        self.assertEqual(wall, 110 + 1000)

    def test_writeback_pool_term_dominates_endpoint_read(self):
        # 写回方向（源 rank 4 → edge 5 单跳）：λ = 100 + 10；资源集 =
        # 源 HBM 读 + NoC + 池端口。源端口在册他流 1 条（u=1）→ 读腿
        # = 1000/(100/2) = 20 < 池项 200（独端口）→ wall = 110 + 200。
        registry = HbmPortFlowRegistry()
        registry.register(4, "bg#flow")
        wall = _pool_transfer_ns_shards(
            _rates(), LinkFlowRegistry(), registry,
            kind="pool_writeback",
            edge_ranks_by_rank=(5,),
            paths_by_rank=((4, 5),),
            bytes_by_rank=(1000,))
        self.assertEqual(wall, 110 + 200)

    def test_writeback_source_hbm_read_enters_max(self):
        # 源 HBM 慢：读腿 = 1000/(1/2) = 2000（u=1，endpoint 查询
        # path[0] 而非 path[-1]）主导 → 写回 wall = 110 + 2000。
        registry = HbmPortFlowRegistry()
        registry.register(4, "bg#flow")
        wall = _pool_transfer_ns_shards(
            _rates(hbm=1.0), LinkFlowRegistry(), registry,
            kind="pool_writeback",
            edge_ranks_by_rank=(5,),
            paths_by_rank=((4, 5),),
            bytes_by_rank=(1000,))
        self.assertEqual(wall, 110 + 2000)

    def test_shared_noc_link_self_overlap_counts(self):
        # 两 shard 路径共链 (0,1)：自身重叠计费（divisor_multi 含自身
        # = 2）→ NoC 项 = 1000/(10/2) = 200 与池项同值（独立端口各
        # 200）→ max = 200；λ 取最长路径 2 跳 = 120。
        wall = _pool_transfer_ns_shards(
            _rates(), LinkFlowRegistry(), None,
            kind="pool_restore",
            edge_ranks_by_rank=(0, 5),
            paths_by_rank=((0, 1), (0, 1, 2)),
            bytes_by_rank=(1000, 1000))
        self.assertEqual(wall, 120 + 200)

    def test_degenerate_no_topology_single_rank_path(self):
        # 退化口径（未注入拓扑 fn 的直接构造夹具）：edge = rank 本地、
        # 单 rank 路径 → 无 NoC 腿/逐跳时延；端点 HBM 腿仍无条件在场
        # （10 < 200 不触 max）→ wall = 池时延 + 池项。
        wall = _pool_transfer_ns_shards(
            _rates(), LinkFlowRegistry(), None,
            kind="pool_restore",
            edge_ranks_by_rank=(7,),
            paths_by_rank=((7,),),
            bytes_by_rank=(1000,))
        self.assertEqual(wall, 100 + 200)

    def test_zero_distance_path_keeps_hbm_endpoint_leg(self):
        # edge == 目标 rank（目标在池端口本地、零 NoC 段）：目标 HBM
        # 写腿不随路径消失——慢 HBM（1 B/ns）下写腿 1000 主导 →
        # wall = 100 + 1000（无逐跳时延）。
        wall = _pool_transfer_ns_shards(
            _rates(hbm=1.0), LinkFlowRegistry(), None,
            kind="pool_restore",
            edge_ranks_by_rank=(3,),
            paths_by_rank=((3,),),
            bytes_by_rank=(1000,))
        self.assertEqual(wall, 100 + 1000)

    def test_zero_bytes_no_transaction(self):
        # 空 shard 契约：零字节无事务——连池端口时延也不收（旧聚合接
        # 口零字节仍收池时延的陷阱废除）。
        for kind in ("pool_restore", "pool_writeback"):
            self.assertEqual(
                _pool_transfer_ns_shards(
                    _rates(), LinkFlowRegistry(), None,
                    kind=kind,
                    edge_ranks_by_rank=(0,),
                    paths_by_rank=((0, 1),),
                    bytes_by_rank=(0,)),
                0)

    def test_contract_validation(self):
        with self.assertRaises(JointCostError):
            _pool_transfer_ns_shards(
                _rates(), LinkFlowRegistry(), None,
                kind="copy",
                edge_ranks_by_rank=(0,), paths_by_rank=((0, 1),),
                bytes_by_rank=(1,))
        with self.assertRaises(JointCostError):
            _pool_transfer_ns_shards(
                _rates(), LinkFlowRegistry(), None,
                kind="pool_restore",
                edge_ranks_by_rank=(0,), paths_by_rank=((0, 1),),
                bytes_by_rank=(1, 2))


# ==================================================== 端到端（估计器）==


class EstimateEndToEndPinTest(unittest.TestCase):
    """估计器入口逐位：copy@remote 恢复走逐 shard 式 (7)（远 shard、
    共享端口、层门联钉）。"""

    def test_copy_remote_history_prep_by_hand(self):
        # 两 rank 同 edge 0（共享端口 own=2 → 池项 400）；rank 3 shard
        # 两跳（λ = 120）→ wall = 120 + 400 = 520。
        model = _model(
            {1: _load(1)},
            nearest_edge=lambda rank: 0,
            rank_route=lambda a, b: {
                (0, 3): (0, 1, 3), (0, 4): (0, 4)}[(a, b)],
            instance_ranks=lambda index: (3, 4))
        candidate = model.estimate_P_with_continuation(
            session=_session(), request=_request(), instance_index=1,
            action="copy", remote_enabled=True)
        self.assertTrue(candidate.applicable)
        self.assertEqual(candidate.breakdown.history_prep_ns, 520)
        self.assertIn("pool_restore_to_target", candidate.breakdown.notes)

    def test_stay_partial_restore_gates_suffix_layers(self):
        # D02 层门 + D04 池腿联钉：PARTIAL 基 stay（驻留目标 + prefix=2
        # 有驻留前缀）恢复 wall = 远实例逐 shard（rank 3 两跳主导）320。
        model = _model({1: _load(1)})
        session = _session(location="partial_hbm_remote",
                           history_bytes=(1000, 1000),
                           missing=(1000, 1000),
                           resident=1, prefix=2)
        candidate = model.estimate_P_with_continuation(
            session=session, request=_request(), instance_index=1,
            action="stay", remote_enabled=True)
        self.assertTrue(candidate.applicable)
        self.assertEqual(candidate.breakdown.history_prep_ns, 100 + 20
                         + 200)


if __name__ == "__main__":
    unittest.main()
