#!/usr/bin/env python3
"""test_hbm_port_flow_registry.py -- C2（WP1b）实例 HBM 端口流注册表 +
u_port 接线 + A4' 补价 rider + E4 去重的零后端单测（F4 口径锚点）。

覆盖（卡 C2 测试清单）：
  1. 登记/注销配对（含失败注入三态：双释放、漏释放、预登记泄漏各一）；
  2. u_port 数值：2 活跃 decode + 1 在册流 → u_port=3、计价除数（+1
     自身）= 4；闲置端口为 0（不假设恒 1）；
  3. breakdown 物化：合成请求 estimate_action 返回的 ActionCostBreakdown
     全 11 字段非 None（允许数值为 0，字段不得缺省）；
  4. A4' 补价 rider（PROVENANCE §20.1）：remote-read 计价含执行端写腿
     ——u_exec=0 时腿值在场、u_exec>0 时按 (u_exec+1) 因子放大并成为
     max 腿；首 credit 腿按 credit_steps 倍乘、其余流送段按
     (read_passes − credit_steps) 倍乘各一例；
  5. E4 去重（A1 勘误）：estimate_action 路径上 divisor_multi 调用次数
     不增——remote/copy 主计价腿族共享一次并集除数计算、breakdown
     contention_divisor 复用（C1 时代 remote=4 次/copy=2 次 → 本卡 1 次）；
  6. SH 接线（__new__ 范式）：_register_transfer_flows 对 noc_migrate
     双端点端口登记 / 池路径（remote_load/remote_store）不进 HBM 端口
     表（F3）/ 释放幂等 / 活跃 decode provider 的 rank→instance 派生。

运行：cd sh_test_mesh/workload/llama2_7b_inference &&
      python3 joint/test_hbm_port_flow_registry.py   （或 pytest 同路径）
"""
import os
import sys
import unittest
from dataclasses import fields
from types import SimpleNamespace

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
for _p in (_HERE, _PARENT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from face_scheduler import (  # noqa: E402
    KVTransfer as _KVTransfer,
    KVTransferShard as _KVTransferShard,
)
from joint.hbm_port_flow_registry import (  # noqa: E402
    HbmPortFlowError,
    HbmPortFlowRegistry,
)
from joint.joint_cost_model import (  # noqa: E402
    ACTION_COPY,
    ACTION_RECOMPUTE,
    ACTION_REMOTE,
    ACTION_STAY,
    ActionCostBreakdown,
    InstanceLoadView,
    JointCostModel,
    JointHardwareRates,
    LinkFlowRegistry,
    RequestView,
    ServiceFactors,
    SessionKVView,
)


# ============================================================ 夹具构造 ==

def _rates(noc=1000.0, hbm=10.0, lat=0):
    return JointHardwareRates.from_gbps(
        noc_link_gbps=noc, pool_port_gbps=5.0, local_hbm_gbps=hbm,
        d2d_latency_ns=lat, pool_latency_ns=100)


def _load(index=0, total=0):
    return InstanceLoadView(
        instance_index=index, queued_task_load_ns=total,
        running_task_load_ns=0, active_decode_task_load_ns=0,
        hbm_remaining_bytes_by_tp_rank=(10**9, 10**9))


def _session(home=0, resident=1, history_bytes=(1000, 1000),
             missing=(0, 0), location="local_hbm", prefix=4,
             history_tokens=100):
    return SessionKVView(
        session_id="s", home_instance=home, resident_instance=resident,
        location=location, history_tokens=history_tokens,
        resident_prefix_layers=prefix,
        history_bytes_by_tp_rank=history_bytes,
        missing_bytes_by_tp_rank=missing)


def _request(input_tokens=50, decode=3, input_bytes=(5000, 5000),
             prefill_scan_passes=None):
    return RequestView(
        request_id="r", session_id="s", input_tokens=input_tokens,
        history_tokens_before=100, estimated_decode_tokens=decode,
        horizon_source="session_online_mean",
        input_kv_bytes_by_tp_rank=input_bytes,
        prefill_scan_passes=prefill_scan_passes)


def _disjoint_paths(source, target):
    """tp=2 逐 rank 不相交路径（rank 对 (2s,2t)/(2s+1,2t+1)，各 1 跳）：
    home 端口 = {0,1}、exec 端口 = {2,3}（source=0/target=1 时）。"""
    return ((2 * source, 2 * target), (2 * source + 1, 2 * target + 1))


def _route(source, target):
    return ((source, target), 1)


def _model(flow_registry=None, port_registry=None, *, rates=None,
           route_paths_fn="disjoint", credit_iters="auto"):
    return JointCostModel(
        rates=rates if rates is not None else _rates(),
        loads={0: _load(0), 1: _load(1)},
        flow_registry=flow_registry if flow_registry is not None
        else LinkFlowRegistry(),
        service_factors=ServiceFactors(),
        prefill_ns_per_token=1.0, decode_ns_per_token=2.0,
        model_layers=4, instance_tp_size=2, route_fn=_route,
        route_paths_fn=(
            _disjoint_paths if route_paths_fn == "disjoint"
            else route_paths_fn),
        hbm_port_registry=port_registry,
        remote_credit_iters=credit_iters)


def _registry_with_exec_contention(exec_active, home_active=0):
    """目标（exec）端口 = rank 0/1（target 实例 0）活跃 decode =
    exec_active 的注册表；源（home）端口 = rank 2/3（resident 实例 1）。"""
    registry = HbmPortFlowRegistry()
    registry.attach_active_decode_provider(
        lambda port: exec_active if port in (0, 1) else home_active)
    return registry


# ==================================================== 1. 登记/注销配对 ==


class RegistrationPairingTest(unittest.TestCase):
    """配对纪律：登记/注销一一对应；失败注入三态各一。"""

    def test_register_unregister_pair(self):
        registry = HbmPortFlowRegistry()
        registry.register(7, "r1#merge")
        self.assertEqual(registry.divisor(7), 1)
        self.assertTrue(registry.has_registrations)
        registry.unregister(7, "r1#merge")
        self.assertEqual(registry.divisor(7), 0)
        self.assertEqual(registry.snapshot(), {})
        self.assertEqual(registry.leaked_owners(), {})

    def test_double_release_fails_closed(self):
        # 失败注入①（双释放）：细粒度 unregister 对未登记（已注销）流
        # fail-closed——与 LinkFlowRegistry.unregister 同款报错语义。
        registry = HbmPortFlowRegistry()
        registry.register(7, "r1#merge")
        registry.unregister(7, "r1#merge")
        with self.assertRaises(HbmPortFlowError):
            registry.unregister(7, "r1#merge")

    def test_release_owner_idempotent_on_empty_owner(self):
        # release_owner 幂等空放（释放四点对无登记 owner 合法）——与
        # LinkFlowRegistry.release_owner 返回 0 同口径；不属双释放。
        registry = HbmPortFlowRegistry()
        registry.register(7, "r1#merge")
        self.assertEqual(registry.release_owner("r1#merge"), 1)
        self.assertEqual(registry.release_owner("r1#merge"), 0)

    def test_leaked_owner_visible_after_settlement(self):
        # 失败注入②（漏释放）：结算边界后 owner 仍在册——snapshot 与
        # leaked_owners 必须暴露（不静默吞掉）。
        registry = HbmPortFlowRegistry()
        registry.register(2, "r1#decode")
        registry.register(3, "r1#decode")
        registry.release_owner("other#merge")
        self.assertEqual(registry.snapshot()[2]["u_port_total"], 1)
        self.assertEqual(registry.leaked_owners(), {"r1#decode": (2, 3)})
        self.assertEqual(registry.release_owner("r1#decode"), 2)
        self.assertEqual(registry.leaked_owners(), {})

    def test_preregistration_leak_detected(self):
        # 失败注入③（预登记泄漏）：准入相预登记两条流、结算只核销一条
        # ——残余登记在 leaked_owners 可见（C8 #readplan 对账核销的
        # 前置能力位）。
        registry = HbmPortFlowRegistry()
        registry.register(2, "r9#readplan")
        registry.register(3, "r9")
        registry.unregister(2, "r9#readplan")
        self.assertEqual(registry.leaked_owners(), {"r9": (3,)})
        self.assertEqual(registry.divisor(2), 0)
        self.assertEqual(registry.divisor(3), 1)

    def test_owner_port_mismatch_fails_closed(self):
        registry = HbmPortFlowRegistry()
        registry.register(7, "r1#merge")
        with self.assertRaises(HbmPortFlowError):
            registry.unregister(8, "r1#merge")


# ==================================================== 2. u_port 数值 ==


class UPortDivisorTest(unittest.TestCase):
    """F4：u_port = 活跃 decode 消费流 + 在册传输/远读流（闲置为 0）。"""

    def test_two_active_decode_plus_one_flow(self):
        # 2 活跃 decode + 1 在册流 → u_port = 3；计价端点腿除数（候选
        # 自身 +1，JCM _shard_leg_ns 显式加）= 4。
        registry = HbmPortFlowRegistry()
        registry.attach_active_decode_provider(lambda port: 2)
        registry.register(5, "r1#decode#1")
        self.assertEqual(registry.divisor(5), 3)
        # （F4：原装饰性断言 assertEqual(registry.divisor(5) + 1, 4)
        # 已删——复审发现；+1 属 JCM _shard_leg_ns 的调用侧语义，
        # 在此对 registry.divisor 重复加一不约束任何生产行为。）
        self.assertEqual(registry.decomposition(5), (2, 1))

    def test_idle_port_is_zero(self):
        # 闲置（无活跃 decode、无在册流）= 0——不假设恒 1。
        registry = HbmPortFlowRegistry()
        registry.attach_active_decode_provider(lambda port: 0)
        self.assertEqual(registry.divisor(99), 0)
        self.assertEqual(registry.decomposition(99), (0, 0))
        self.assertEqual(registry.snapshot(), {})

    def test_snapshot_field_names_match_c5_port_schema(self):
        # snapshot 字段名与 C5 冻结的 port_snapshot 逐实例分解字段同名
        #（C11 步骤 6 接通时零改名）。
        registry = HbmPortFlowRegistry()
        registry.attach_active_decode_provider(lambda port: 1)
        registry.register(5, "r1")
        self.assertEqual(
            registry.snapshot()[5],
            {"u_port_active_decode_streams": 1,
             "u_port_registered_transfer_flows": 1,
             "u_port_total": 2})


# ==================================================== 3. breakdown 物化 ==


class BreakdownMaterializationTest(unittest.TestCase):
    """estimate_action 返回的全 11 字段非 None（值可为 0，不得缺省）。"""

    FIELD_NAMES = tuple(field.name for field in fields(ActionCostBreakdown))

    def test_all_eleven_fields_materialized_per_action(self):
        self.assertEqual(len(self.FIELD_NAMES), 11)
        port_registry = _registry_with_exec_contention(exec_active=2)
        model = _model(port_registry=port_registry)
        fixtures = [
            (ACTION_STAY, _session(resident=1), 1),
            (ACTION_RECOMPUTE, _session(resident=1), 0),
            (ACTION_COPY, _session(resident=1), 0),
            (ACTION_REMOTE, _session(resident=1), 0),
        ]
        for action, session, target in fixtures:
            candidate = model.estimate_action(
                session=session, request=_request(),
                instance_index=target, action=action, remote_enabled=True)
            self.assertTrue(candidate.applicable, action)
            self.assertIsNotNone(candidate.breakdown, action)
            for name in self.FIELD_NAMES:
                value = getattr(candidate.breakdown, name)
                self.assertIsNotNone(value, (action, name))
            self.assertIsInstance(candidate.breakdown.notes, tuple)

    def test_u_port_decomposition_in_notes_channel(self):
        # C2 步骤 3：u_port 分解先走决策日志 notes 通道（C5 冻结 schema
        # 不动；port_snapshot 正式字段位归 C11）。路由方向：source =
        # resident（实例 1，rank 2/3）→ home 读腿端口；target = 实例 0
        # （rank 0/1）→ exec 写腿端口（活跃 decode = 2 披露在 exec 侧）。
        port_registry = _registry_with_exec_contention(exec_active=2)
        model = _model(port_registry=port_registry)
        candidate = model.estimate_action(
            session=_session(resident=1), request=_request(),
            instance_index=0, action=ACTION_REMOTE, remote_enabled=True)
        self.assertIn("u_port_home=r2:0act+0fl",
                      candidate.breakdown.notes)
        self.assertIn("u_port_exec=r0:2act+0fl",
                      candidate.breakdown.notes)

    def test_no_u_port_note_without_registry(self):
        # 离线/单测口径（未注入注册表）：不披露（与 C1 行为零差异）。
        model = _model()
        candidate = model.estimate_action(
            session=_session(resident=1), request=_request(),
            instance_index=0, action=ACTION_REMOTE, remote_enabled=True)
        self.assertFalse([
            note for note in candidate.breakdown.notes
            if note.startswith("u_port_")])


# ============================================ 4. A4' 补价 rider（§20.1）==


class RemoteReadExecWriteLegTest(unittest.TestCase):
    """kind="read" 增执行端 HBM 写腿（与 copy/staging 第三腿同式同源）。"""

    # hbm=10 慢端点、noc=1000 快链路 → 端点腿主导；B=1000、passes=4。
    BYTES_PER_RANK = 1000

    def _remote_read_ns(self, port_registry):
        model = _model(
            rates=_rates(noc=1000.0, hbm=10.0, lat=0),
            port_registry=port_registry)
        candidate = model.estimate_action(
            session=_session(history_bytes=(self.BYTES_PER_RANK,
                                            self.BYTES_PER_RANK)),
            request=_request(decode=3),  # passes = 1 prefill + 3 decode = 4
            instance_index=0, action=ACTION_REMOTE, remote_enabled=True)
        return candidate

    def test_exec_write_leg_value_in_place_at_zero_u_exec(self):
        # u_exec=0：执行端写腿在场（与 home 读腿同率同值——无争用时两
        # 端点腿并列 max）。B×passes/hbm = 1000×4/10 = 400。
        candidate = self._remote_read_ns(
            _registry_with_exec_contention(exec_active=0))
        self.assertEqual(
            candidate.breakdown.remote_read_ns,
            self.BYTES_PER_RANK * 4 // 10)

    def test_exec_write_leg_amplifies_and_dominates(self):
        # u_exec=3（执行端与 COMPUTE/增量写 N-way 均分）：写腿按
        # (u_exec+1)=4 因子放大 = 1000×4×4/10 = 1600 > home 腿 400，
        # 成为 max 腿（整腿 wall 随之抬升——补价方向：只高不低）。
        quiet = self._remote_read_ns(
            _registry_with_exec_contention(exec_active=0))
        contended = self._remote_read_ns(
            _registry_with_exec_contention(exec_active=3))
        self.assertEqual(contended.breakdown.remote_read_ns, 1600)
        self.assertGreater(
            contended.breakdown.remote_read_ns,
            quiet.breakdown.remote_read_ns)
        self.assertGreater(
            contended.cost_ns, quiet.cost_ns)

    def test_first_credit_and_stream_multiplier_split(self):
        # 倍乘切法（与现有字节基数同切法）：passes=4、auto K=ceil(4/8)=1
        # → 首 credit 腿按 credit_steps=1 倍乘 = 1000×1×4/10 = 400；
        # 其余流送段按 (read_passes−credit_steps)=3 倍乘 = 1200；全流
        # = 1600（remote_read_ns 恒为全流总时延口径）。
        candidate = self._remote_read_ns(
            _registry_with_exec_contention(exec_active=3))
        breakdown = candidate.breakdown
        self.assertIn("remote_credit_k=1", breakdown.notes)
        self.assertEqual(breakdown.remote_read_first_credit_ns, 400)
        self.assertEqual(breakdown.remote_read_stream_ns, 1200)
        self.assertEqual(breakdown.remote_read_ns, 1600)

    def test_capacity_half_side_unchanged(self):
        # F14 容量半边不动：u_exec 争用只放大带宽腿，不改变空间足迹
        # （remote-read 足迹仍 = input 增量，两争用态同值）。
        quiet = self._remote_read_ns(
            _registry_with_exec_contention(exec_active=0))
        contended = self._remote_read_ns(
            _registry_with_exec_contention(exec_active=3))
        self.assertEqual(
            quiet.breakdown.eviction_wait_ns,
            contended.breakdown.eviction_wait_ns)


# ==================================================== 5. E4 去重（A1）==


class _CountingFlows(LinkFlowRegistry):
    """divisor_multi 调用计数探针（E4 去重断言用）。"""

    def __init__(self):
        super().__init__()
        self.multi_calls = 0

    def divisor_multi(self, paths, *, include_self=True):
        self.multi_calls += 1
        return super().divisor_multi(paths, include_self=include_self)


class E4DivisorDedupTest(unittest.TestCase):
    """estimate_action 路径上 divisor_multi 调用次数不增（remote/copy
    严格下降）：主计价腿族共享一次并集除数、breakdown 复用之。"""

    def _flows_and_model(self):
        flows = _CountingFlows()
        return flows, _model(flow_registry=flows)

    def test_remote_estimate_calls_divisor_multi_once(self):
        # home==target（merge 跳过）+ LOCAL 基：读流族（全流/首 credit/
        # 其余流送）共享一次并集除数、breakdown 零额外调用 → 1 次。
        # C1 时代同夹具 = 4 次（3 腿 + breakdown 披露第二次）。
        flows, model = self._flows_and_model()
        candidate = model.estimate_action(
            session=_session(home=0, resident=1), request=_request(),
            instance_index=0, action=ACTION_REMOTE, remote_enabled=True)
        self.assertTrue(candidate.applicable)
        self.assertEqual(flows.multi_calls, 1)

    def test_copy_estimate_calls_divisor_multi_once(self):
        # C1 时代同夹具 = 2 次（copy 腿 + breakdown）→ 本卡 1 次。
        flows, model = self._flows_and_model()
        candidate = model.estimate_action(
            session=_session(home=0, resident=1), request=_request(),
            instance_index=0, action=ACTION_COPY, remote_enabled=True)
        self.assertTrue(candidate.applicable)
        self.assertEqual(flows.multi_calls, 1)

    def test_stay_recompute_keep_single_call(self):
        # 无 NoC 计价腿可复用 → 函数尾单次调用（C1 时代也恰一次——
        # "不增"断言的留守分支；不得为 0：contention_divisor 仍须物化）。
        flows, model = self._flows_and_model()
        model.estimate_action(
            session=_session(home=0, resident=1), request=_request(),
            instance_index=1, action=ACTION_STAY, remote_enabled=True)
        self.assertEqual(flows.multi_calls, 1)
        flows2, model2 = self._flows_and_model()
        model2.estimate_action(
            session=_session(home=0, resident=1), request=_request(),
            instance_index=0, action=ACTION_RECOMPUTE, remote_enabled=True)
        self.assertEqual(flows2.multi_calls, 1)

    def test_contention_divisor_reuses_priced_value(self):
        # 复用值正确性：背景流登记后，breakdown.contention_divisor 等于
        # 主计价腿算得的并集除数（含自身与他流）。路由 source=resident
        # 实例 1 → rank 2，背景流登记在有向链路 (2,0) 上。
        flows = _CountingFlows()
        flows.register_path((2, 0), owner="bg#1")
        model = _model(flow_registry=flows)
        candidate = model.estimate_action(
            session=_session(home=0, resident=1), request=_request(),
            instance_index=0, action=ACTION_REMOTE, remote_enabled=True)
        self.assertEqual(candidate.breakdown.contention_divisor, 2)


# ==================================================== 6. SH 接线 ==


def _shard(source_rank, target_rank, *, edge_rank=None, path=None,
           total_bytes=100):
    return _KVTransferShard(
        source_rank=source_rank, target_rank=target_rank,
        edge_rank=edge_rank, bytes=total_bytes,
        noc_path=path if path is not None else (source_rank, target_rank),
        layer_start=0, layer_end=2)


def _transfer(kind, shards, *, total_bytes=None):
    return _KVTransfer(
        kind=kind, phase="prefill", reason="joint_test",
        session_id="s", trigger_request_id="r1",
        source_instance_index=0, target_instance_index=1,
        total_bytes=(total_bytes if total_bytes is not None else sum(
            shard.bytes for shard in shards)),
        shards=tuple(shards), model_layers=2, layer_start=0, layer_end=2,
        resident_prefix_layers_before=2, resident_prefix_layers_after=2)


class _ShPortWiringTest(unittest.TestCase):
    """__new__ 范式：_register/_release_transfer_flows 的 HBM 端口侧。"""

    def _scheduler(self):
        from online.sh30_online_scheduler import Sh30OnlineScheduler
        scheduler = Sh30OnlineScheduler.__new__(Sh30OnlineScheduler)
        scheduler._joint_flows = LinkFlowRegistry()
        scheduler._pool_ports = SimpleNamespace(
            register=lambda *args, **kwargs: None,
            release_owner=lambda owner: 0)
        scheduler._hbm_ports = HbmPortFlowRegistry()
        return scheduler

    def test_noc_migrate_registers_both_endpoint_ports(self):
        # home 读腿恒有 + exec 写腿（copy/merge/remote-read 同登记型）：
        # 逐 shard 双端点端口在册；中间跳（无——单跳夹具）不涉及。
        scheduler = self._scheduler()
        scheduler._register_transfer_flows(
            [_transfer("noc_migrate", [_shard(0, 2), _shard(1, 3)])],
            owner="r1")
        self.assertEqual(scheduler._hbm_ports.divisor(0), 1)
        self.assertEqual(scheduler._hbm_ports.divisor(2), 1)
        self.assertEqual(scheduler._hbm_ports.divisor(1), 1)
        self.assertEqual(scheduler._hbm_ports.divisor(3), 1)
        self.assertEqual(
            scheduler._hbm_ports.leaked_owners(),
            {"r1": (0, 2, 1, 3)})

    def test_pool_transfers_do_not_enter_hbm_ports(self):
        # F3 边界：remote_store/remote_load 走池端口口径，不进 HBM 端
        # 口表（本地链路段也不登记——池路径 NoC 腿未计价的既定披露）。
        scheduler = self._scheduler()
        scheduler._register_transfer_flows(
            [_transfer("remote_store", [_shard(0, None, edge_rank=4,
                                               path=(0, 4))])],
            owner="r1#decode")
        self.assertEqual(scheduler._hbm_ports.snapshot(), {})
        self.assertEqual(scheduler._hbm_ports.divisor(0), 0)

    def test_release_clears_and_is_idempotent(self):
        scheduler = self._scheduler()
        scheduler._register_transfer_flows(
            [_transfer("noc_migrate", [_shard(0, 2)])], owner="r1#merge")
        scheduler._release_transfer_flows("r1#merge")
        self.assertEqual(scheduler._hbm_ports.snapshot(), {})
        # 幂等空放（释放四点对无登记 owner 合法）。
        scheduler._release_transfer_flows("r1#merge")

    def test_active_decode_provider_rank_to_instance(self):
        # F4：活跃 decode 消费流 = 端口所属实例的 active_decode 现值
        #（TP 全 rank 同账）；闲实例为 0。
        from online.sh30_online_scheduler import Sh30OnlineScheduler
        scheduler = Sh30OnlineScheduler.__new__(Sh30OnlineScheduler)
        scheduler._rank_to_instance = {0: 0, 1: 0, 2: 1, 3: 1}
        scheduler.instances = [
            SimpleNamespace(active_decode=["a", "b"]),
            SimpleNamespace(active_decode=[]),
        ]
        self.assertEqual(scheduler._hbm_active_decode_streams(0), 2)
        self.assertEqual(scheduler._hbm_active_decode_streams(1), 2)
        self.assertEqual(scheduler._hbm_active_decode_streams(3), 0)
        self.assertEqual(scheduler._hbm_active_decode_streams(99), 0)


if __name__ == "__main__":
    unittest.main()
