#!/usr/bin/env python3
"""test_joint_fixes.py -- R1'-R15 修复批的关键语义单元测试（2026-09-14）。

覆盖（对应 kimi 终稿 §5 用例的核心可单测断言）：
  - R1'：动作感知预约足迹单源 + PARTIAL 去钉扎（可行性与物理可行性
    不再被驻留亲和掩码）；remote-read 足迹 = input 增量（F1 消灭）。
  - N1(a)：PARTIAL 会话的 remote-read 适用性收窄（后端能力边界）。
  - R13/N9：recompute@home 缺失后缀物化、无工作副本（stay 同构本地
    事务）、merge 零搬运；跨实例 recompute 工作副本语义不变。
  - R4：merge home 容量不足 → 基础前缀自降级（merge_degrade_events
    落账）→ k=0 落 REMOTE 归并（永不失败）；重复 merge 版本键。
  - N3'：容量类异常类型化（KVCapacityError 携带已提交逐出）；合同类
    异常不被类型化吞掉。
  - M3：跨实例 decode 预约移动 fail-closed。
  - R14（调度器侧）：停滞登记/唤醒键/死锁守卫（__new__ 壳夹具）。
  - R3'：驱逐等待定价区分逐出可解/深缺口（活跃剩余负载峰值）。
  - R15-2/P3：ServiceFactors α 时间衰减公式 + 同时刻汇总 + 拒绝样本。
  - R15-1/F-B：divisor_multi 按全部 TP 并行流链路并集取瓶颈。
  - R12：水印重放的 joint 双驻留原语（stash_base / merge 结算）。

运行：cd sh_test_mesh/workload/llama2_7b_inference &&
      python3 joint/test_joint_fixes.py   （或 pytest 同路径）
"""
import os
import sys
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_WL = os.path.dirname(_HERE)
for _p in (_HERE, _WL):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from face_scheduler import (  # noqa: E402
    KVCapacityError,
    KVCacheManager,
    kv_cache_shard_bytes_for_tokens,
)
from joint.test_joint_mechanisms import (  # noqa: E402
    _seed,
    _tiny_hardware,
    _tiny_model,
    _two_instance_topology,
)
from joint.joint_cost_model import (  # noqa: E402
    InstanceLoadView,
    JointCostModel,
    JointHardwareRates,
    LinkFlowRegistry,
    ServiceFactors,
    SessionKVView,
    RequestView,
    _pool_transfer_ns,
)


# ===================================================== R1'/N1/N3'/R4/R13 ==

def _manager(capacity_bytes=1_000_000_000, **kwargs):
    hardware = _tiny_hardware(capacity_bytes)
    topology = _two_instance_topology(hardware)
    return KVCacheManager(
        topology, _tiny_model(4),
        category_mode="typed", layer_policy="minimal_layer_groups",
        **kwargs)


def _make_partial(kv, session_id="s", tokens=10, completion_ns=10):
    """在实例 0 建立 PARTIAL 会话（half 后缀逐出）。"""
    _seed(kv, session_id, 0, tokens, completion_ns, "human")
    kv._evict_suffix(
        kv._sessions[session_id],
        phase="history", reason="test_partial",
        trigger_request_id="seed")
    return kv._sessions[session_id]


class FootprintSingleSourceTest(unittest.TestCase):
    """R1'：joint_reservation_context_tokens 单源 + 去钉扎。"""

    def test_basis_by_action(self):
        kv = _manager()
        self.assertEqual(
            kv.joint_reservation_context_tokens(
                action=None, history_tokens=100, input_tokens=5),
            105)
        for action in ("stay", "copy", "recompute"):
            self.assertEqual(
                kv.joint_reservation_context_tokens(
                    action=action, history_tokens=100, input_tokens=5),
                105)
        self.assertEqual(
            kv.joint_reservation_context_tokens(
                action="remote-read", history_tokens=100, input_tokens=5),
            5)
        with self.assertRaises(ValueError):
            kv.joint_reservation_context_tokens(
                action="bogus", history_tokens=1, input_tokens=1)

    def test_partial_depinning_allows_other_instances(self):
        """PARTIAL 会话不再构成实例亲和掩码：容量允许时其他实例可行。"""
        kv = _manager()
        _make_partial(kv)
        feasible = kv.request_hbm_feasible_instances(
            session_id="s", final_context_tokens=20)
        self.assertEqual(feasible, (True, True))
        eventual = kv.request_hbm_eventually_feasible_instances(
            session_id="s", final_context_tokens=20)
        self.assertEqual(eventual, (True, True))

    def test_remote_read_footprint_is_input_only(self):
        """F1 消灭：remote-read 只预约 input 增量（整份放不下但 input
        放得下 → 物理可行；eventually 同口径）。tiny model：256 B/token
        全层、128 B/token/rank；权重 11,392 B/rank。"""
        kv = _manager(capacity_bytes=100_000)
        _seed(kv, "s", 0, 400, 10, "human")  # LOCAL 基础在实例 0
        stay = kv.request_hbm_eventually_feasible_instances(
            session_id="s", final_context_tokens=800, action="stay")
        remote = kv.request_hbm_eventually_feasible_instances(
            session_id="s", final_context_tokens=800,
            action="remote-read")
        # 整份 800（≈102 KB/rank > free ≈88 KB）放不下任一实例；
        # input 增量 400（≈51 KB/rank）在两实例都放得下。
        self.assertEqual(stay, (False, False))
        self.assertEqual(remote, (True, True))


class RecomputeAtHomeSemanticsTest(unittest.TestCase):
    """R13/N9：recompute@home = 驻留前缀保持权威 + 缺失后缀物化。"""

    def test_recompute_at_partial_home_materializes_suffix_only(self):
        kv = _manager()
        session = _make_partial(kv, tokens=10)
        prefix_before = session.resident_prefix_layers
        before = kv.session_snapshot("s")
        _before, transfers, evictions = kv.prepare_prefill(
            session_id="s", target_instance_index=0,
            history_tokens=10, trigger_request_id="t1",
            action="recompute")
        # 无传输（重算物化 = 计算量，非搬运）；无工作副本（stay 同构）。
        self.assertEqual(transfers, ())
        after = kv.session_snapshot("s")
        self.assertIsNone(after.working_kind)
        self.assertEqual(after.location, kv.LOCAL_HBM)
        self.assertEqual(after.resident_prefix_layers, 4)  # 全层驻留
        # 物化字节 = 缺失后缀层（prefix_before..L），不多算前缀。
        suffix = kv_cache_shard_bytes_for_tokens(
            kv.model, 10, kv.tp_degree)
        suffix_only = suffix[0] * (4 - prefix_before) // 4
        used0 = kv.hbm_snapshots(0)[0].kv_cache_bytes
        used0_before = before.local_bytes // kv.tp_degree
        self.assertAlmostEqual(
            used0 - used0_before, suffix_only, delta=2)
        # merge：stay 等价零流量（无工作副本）。
        kv.expand_prefill(
            session_id="s", instance_index=0, context_tokens=14,
            trigger_request_id="t1")
        self.assertEqual(
            kv.merge_back(session_id="s", trigger_request_id="t1",
                          new_tokens=4), ())
        kv.mark_complete("s", 20, "human")

    def test_cross_instance_recompute_keeps_working_copy(self):
        kv = _manager()
        _seed(kv, "s", 0, 10, 10, "human")
        kv.prepare_prefill(
            session_id="s", target_instance_index=1,
            history_tokens=10, trigger_request_id="t1",
            action="recompute")
        working = kv.session_snapshot("s")
        self.assertEqual(working.working_kind, "recompute")
        self.assertEqual(working.context_tokens, 0)
        kv.expand_prefill(
            session_id="s", instance_index=1, context_tokens=13,
            trigger_request_id="t1")
        transfers = kv.merge_back(
            session_id="s", trigger_request_id="t1", new_tokens=3)
        merged = kv.session_snapshot("s")
        self.assertEqual(merged.instance_index, 0)
        self.assertEqual(merged.context_tokens, 13)
        self.assertTrue(all(t.kind != "local_hit" for t in transfers))


class MergeSelfDegradeTest(unittest.TestCase):
    """R4：merge home 容量不足 → 基础前缀自降级（红线 1 独立事务）。"""

    def test_degrade_settles_partial_with_event_ledger(self):
        # home 灌满活跃占用 → merge 空间不足 → 基础前缀自降级（红线 1
        # 独立事务）→ 降级事件落账、结算落在降级后的 PARTIAL 前缀。
        # tiny model：128 B/token/rank；权重 11,392 B/rank。
        kv = _manager(capacity_bytes=51_328)
        _seed(kv, "s", 0, 10, 10, "human")
        _seed(kv, "hog", 0, 300, 10, "human")
        kv._sessions["hog"].active = True  # 不可逐的活跃占用
        kv.prepare_prefill(
            session_id="s", target_instance_index=1,
            history_tokens=10, trigger_request_id="t1", action="copy")
        kv.expand_prefill(
            session_id="s", instance_index=1, context_tokens=14,
            trigger_request_id="t1")
        transfers = kv.merge_back(
            session_id="s", trigger_request_id="t1", new_tokens=4)
        self.assertTrue(kv.merge_degrade_events, "自降级事件必须落账")
        reasons = {t.reason for t in transfers}
        self.assertIn("home_merge_base_degrade", reasons)
        merged = kv.session_snapshot("s")
        # 降级后结算：home 侧 PARTIAL（前缀 < 全层），非幻影全层驻留。
        self.assertEqual(merged.location, kv.PARTIAL_HBM_REMOTE)
        self.assertEqual(merged.instance_index, 0)
        self.assertLess(merged.resident_prefix_layers, 4)
        kv.mark_complete("s", 20, "human")

    def test_duplicate_merge_fails_closed(self):
        kv = _manager()
        _seed(kv, "s", 0, 10, 10, "human")
        kv.prepare_prefill(
            session_id="s", target_instance_index=1,
            history_tokens=10, trigger_request_id="t1", action="copy")
        kv.merge_back(session_id="s", trigger_request_id="t1", new_tokens=3)
        with self.assertRaises(RuntimeError):
            kv.merge_back(
                session_id="s", trigger_request_id="t1", new_tokens=3)


class TypedCapacityErrorsTest(unittest.TestCase):
    """N3'：容量类类型化；合同类异常原样上抛。"""

    def test_deep_gap_raises_typed_capacity_error(self):
        kv = _manager(capacity_bytes=51_328)
        _seed(kv, "s", 0, 10, 10, "human")
        _seed(kv, "hog", 0, 300, 10, "human")
        kv._sessions["hog"].active = True
        with self.assertRaises(KVCapacityError):
            kv.reserve_request_capacity(
                request_id="r9", session_id="s",
                instance_index=0, final_context_tokens=90)

    def test_contract_duplicate_reservation_not_swallowed(self):
        kv = _manager()
        kv.reserve_request_capacity(
            request_id="r1", session_id="s1", instance_index=0,
            final_context_tokens=10)
        with self.assertRaises(ValueError) as ctx:
            kv.reserve_request_capacity(
                request_id="r1", session_id="s1", instance_index=0,
                final_context_tokens=10)
        self.assertNotIsInstance(ctx.exception, KVCapacityError)

    def test_m3_cross_instance_reservation_move_fails_closed(self):
        kv = _manager()
        kv.reserve_request_capacity(
            request_id="r1", session_id="s1", instance_index=0,
            final_context_tokens=10)
        with self.assertRaises(RuntimeError):
            kv.move_request_capacity_reservation(
                request_id="r1", target_instance_index=1)


# ============================================================ R15 模型 ==

def _model(loads, flows=None, factors=None):
    rates = JointHardwareRates.from_gbps(
        noc_link_gbps=10.0, pool_port_gbps=5.0, local_hbm_gbps=100.0,
        d2d_latency_ns=10, pool_latency_ns=100)
    return JointCostModel(
        rates=rates, loads=loads,
        flow_registry=flows or LinkFlowRegistry(),
        service_factors=factors or ServiceFactors(),
        prefill_ns_per_token=1.0, decode_ns_per_token=2.0,
        model_layers=4, instance_tp_size=2, route_fn=_route2)


def _route2(source, target):
    return ((source, target), 1)


def _load(remaining=(10**9, 10**9), reclaimable=None, active_ns=0):
    return InstanceLoadView(
        instance_index=0, queued_task_load_ns=0,
        running_task_load_ns=active_ns, active_decode_task_load_ns=0,
        hbm_remaining_bytes_by_tp_rank=remaining,
        reclaimable_bytes_by_tp_rank=(
            reclaimable if reclaimable is not None else remaining))


class EvictionWaitPricingTest(unittest.TestCase):
    """R3'：深缺口按活跃剩余负载峰值计价。"""

    def test_deep_gap_uses_active_load(self):
        model = _model(_load(remaining=(100, 100), reclaimable=(100, 100),
                             active_ns=10**7))
        notes = []
        wait = model._eviction_wait_estimate(
            _load(remaining=(100, 100), reclaimable=(100, 100),
                  active_ns=10**7),
            (500, 500), notes, instance_index=0)
        self.assertGreaterEqual(wait, 10**7)
        self.assertIn("execution_growth_deep_gap_unresolved", notes)

    def test_shallow_gap_uses_pool_writeback(self):
        load = _load(remaining=(100, 100), reclaimable=(10**9, 10**9),
                     active_ns=10**9)
        model = _model(_load())
        notes = []
        wait = model._eviction_wait_estimate(
            load, (500, 500), notes, instance_index=0)
        self.assertEqual(
            wait,
            _pool_transfer_ns(total_bytes=400, divisor=1,
                              rates=model.rates))
        self.assertNotIn("execution_growth_deep_gap_unresolved", notes)


class ServiceFactorsAlphaTest(unittest.TestCase):
    """R15-2/P3：α = 1 − exp(−Δt/τ)；同时刻汇总；非法样本拒绝。"""

    def test_time_decay_and_first_sample(self):
        factors = ServiceFactors()
        factors.observe_prefill(
            actual_ns=100, base_ns=100, service_ns=100, tick_ns=1000)
        factors.flush()
        self.assertEqual(factors.prefill_factor, 1.0)  # 首样本直接初始化
        factors.observe_prefill(
            actual_ns=300, base_ns=100, service_ns=100, tick_ns=2000)
        factors.flush()
        # Δt=1000, τ=100 → α = 1-exp(-10) ≈ 0.9999546；
        # value = (1-α)·1.0 + α·3.0 ≈ 2.99991。
        alpha = 1 - 4.5399929762484854e-5
        expected = (1 - alpha) * 1.0 + alpha * 3.0
        self.assertAlmostEqual(factors.prefill_factor, expected, places=9)

    def test_same_tick_samples_aggregate(self):
        factors = ServiceFactors()
        factors.observe_decode(
            actual_ns=100, base_ns=100, service_ns=100, tick_ns=5)
        factors.observe_decode(
            actual_ns=300, base_ns=100, service_ns=100, tick_ns=5)
        factors.flush()
        # 同时刻先汇总：Σactual/Σbase = 400/200 = 2.0（一条更新）。
        self.assertEqual(factors.decode_factor, 2.0)
        self.assertEqual(factors.updates["decode_factor"], 1)

    def test_invalid_samples_rejected(self):
        factors = ServiceFactors()
        factors.observe_prefill(
            actual_ns=0, base_ns=100, service_ns=100, tick_ns=1)
        factors.observe_prefill(
            actual_ns=100, base_ns=0, service_ns=100, tick_ns=1)
        self.assertEqual(factors.rejected["prefill_factor"], 2)
        self.assertEqual(factors.prefill_factor, 1.0)


class DivisorMultiTest(unittest.TestCase):
    """R15-1/F-B：非代表路径上的争用被计价（并集瓶颈）。"""

    def test_non_representative_path_contention_counted(self):
        registry = LinkFlowRegistry()
        # 代表对路径 (0->2)；非代表 TP shard 路径 (1->3) 有他流登记。
        registry.register(1, 3)
        divisor = registry.divisor_multi(
            ((0, 2), (1, 3)), include_self=True)
        self.assertEqual(divisor, 2)

    def test_self_multiplicity_on_shared_link(self):
        registry = LinkFlowRegistry()
        # 候选自身两条 TP 路径共享链路 (4,5)：自身份额 = 2。
        divisor = registry.divisor_multi(
            ((4, 5), (4, 5)), include_self=True)
        self.assertEqual(divisor, 2)

    def test_owner_release(self):
        registry = LinkFlowRegistry()
        flow_id = registry.register_path((0, 1, 2), owner="r1#decode")
        self.assertEqual(registry.divisor((0, 1), include_self=False), 1)
        self.assertEqual(
            registry.divisor((0, 1), include_self=True), 2)
        registry.release_owner("r1#decode")
        self.assertEqual(registry.snapshot(), {})
        with self.assertRaises(Exception):
            registry.release_flow(flow_id)


# ======================================================== R14 调度器侧 ==

class DecodeStallMachineryTest(unittest.TestCase):
    """R14：停滞登记 / 唤醒键 / 死锁守卫（__new__ 壳夹具）。"""

    def _scheduler(self):
        from online.sh30_online_scheduler import (
            Sh30OnlineScheduler, _OnlineInstanceState, _OnlineRequestRuntime,
        )
        scheduler = Sh30OnlineScheduler.__new__(Sh30OnlineScheduler)
        scheduler.instances = [_OnlineInstanceState(index=0),
                               _OnlineInstanceState(index=1)]
        scheduler.kv_manager = _manager()
        scheduler._kv_ledger_epoch = 0
        scheduler._stalled_by_instance = {}
        scheduler._batch = None
        scheduler.online_log_count = 0
        scheduler.decision_log_sink = None
        scheduler.online_log_rows = []
        # 自查 A 后守卫引用的跨 tick 事件源状态（空 = 无未来事件源，
        # 守卫可在全停滞时触发）。
        scheduler._pending_merge_alarms = {}
        scheduler.arrival_heap = []
        scheduler.runtimes = []
        scheduler._arrived_request_count = 0
        runtime = _OnlineRequestRuntime({
            "request_id": "r0", "session_id": "s0", "turn_index": 0,
            "queue_index": 0, "prefill_length": 512, "decode_length": 8,
            "history_tokens_before": 0,
            "prefill_context_tokens": 512,
            "final_context_tokens": 520,
        }, 512)
        runtime.decode_instance_index = 0
        return scheduler, runtime

    def test_stall_wake_and_deadlock_guard(self):
        scheduler, runtime = self._scheduler()
        # 会话在 kv 账本就位（wake 重试读真实 manager）。
        scheduler.kv_manager.prepare_prefill(
            session_id="s0", target_instance_index=0,
            history_tokens=0, trigger_request_id="r0")
        scheduler.kv_manager.expand_prefill(
            session_id="s0", instance_index=0, context_tokens=512,
            trigger_request_id="r0")
        scheduler._enter_decode_stall(runtime, 0, "insufficient HBM")
        self.assertTrue(runtime.decode_stalled)
        self.assertIn(0, scheduler._stalled_by_instance)
        # 唤醒键：KV 纪元未变且实例纪元未变 → 不重试（无自旋）。
        key_before = runtime.stall_wake_key
        scheduler._wake_stalled_decodes(0)
        self.assertEqual(runtime.stall_wake_key, key_before)
        self.assertTrue(runtime.decode_stalled)
        # 容量释放（KV 纪元 bump）→ 重试窗口打开（会话上下文已就位则
        # 直接唤醒退出停滞）。
        scheduler._bump_kv_ledger_epoch()
        scheduler._wake_stalled_decodes(0)
        self.assertFalse(runtime.decode_stalled)
        self.assertNotIn(0, scheduler._stalled_by_instance)
        # 重新停滞且无任何事件源 → 死锁守卫显式 fail-closed。
        scheduler.kv_manager._sessions["s0"].context_tokens = 0
        runtime.decode_tokens_consumed = 1
        scheduler._enter_decode_stall(runtime, 0, "insufficient HBM")
        scheduler.instances[0].active_decode.append(runtime)
        with self.assertRaises(RuntimeError) as ctx:
            scheduler._check_decode_deadlock()
        self.assertIn("deadlock", str(ctx.exception))

    def test_guard_passes_when_work_remains(self):
        scheduler, runtime = self._scheduler()
        scheduler.instances[0].qp.append(runtime)
        runtime.remaining_chunks = 2
        scheduler._enter_decode_stall(runtime, 0, "insufficient HBM")
        # qp 仍有 chunk 工作 → 有事件源，守卫不触发。
        scheduler._check_decode_deadlock()


# =========================================================== R12 水印 ==

class WatermarkJointReplayTest(unittest.TestCase):
    """R12：joint 双驻留重放原语（stash/merge 结算）。"""

    def _replay(self):
        sys.path.insert(0, os.path.join(_WL, "..", "..", "slo_tools"))
        import hbm_watermark
        mapping = hbm_watermark.REPO_VARIANTS["astra-sim-joint"]
        return hbm_watermark.WatermarkReplay(
            "astra-sim-joint", mapping,
            {"requests": {}, "path": "test"}, coef_bytes_per_token=100,
            capacity_per_instance=None)

    def test_stash_and_merge_settle(self):
        replay = self._replay()
        # 实例 0 驻留 1000B → 跨实例工作副本（实例 1 搬入 600B）。
        replay.apply_grow(1, "s", 0, 10, "prefill_grow")
        replay.stash_base("s")
        session = replay.sessions["s"]
        self.assertEqual((session.base_instance, session.base_bytes), (0, 1000))
        self.assertIsNone(session.instance)
        # merge：noc 增量 200B 回 home；执行端释放。
        replay._apply(2, 1, 600)
        session.bytes = 600
        session.instance = 1
        self.assertEqual(replay.occupancy, {0: 1000, 1: 600})
        # 模拟 _joint_completion 的核心结算序。
        replay._apply(3, 1, -600)
        session.bytes = 0
        session.instance = None
        replay._apply(3, 0, 200)  # home 侧只加增量（base 未离开）
        session.instance = 0
        session.bytes = 1200
        session.base_instance = None
        session.base_bytes = 0
        self.assertEqual(replay.occupancy, {0: 1200, 1: 0})

    def test_double_stash_fails_closed(self):
        replay = self._replay()
        replay.apply_grow(1, "s", 0, 10, "prefill_grow")
        replay.stash_base("s")
        import hbm_watermark
        with self.assertRaises(hbm_watermark.SloToolError):
            replay.stash_base("s")


if __name__ == "__main__":
    unittest.main()
