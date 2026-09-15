#!/usr/bin/env python3
"""test_joint_review2_fixes.py -- kimi 复审+终审修复批定向测试
（2026-09-14 第二轮 / 2026-09-15 终审-中1..中5 与四审低项）。

覆盖（对应复审报告 K1-K6 + M1/M2/M3/M5/M6/覆盖缺口）：
  - K1：死锁守卫逃逸条件含 pending_decode_ready（drain→加入列车窗口
    是活跃事件源，不得误杀合法 run）。
  - K2：水印完成结算只并入增量（base 从未离开 home，不得双计复利）
    ——驱动**真实** WatermarkScan.consume 路径（非手工模拟结算序）。
  - K3：REMOTE 基会话执行端工作副本在完成时释放（按"是否存在工作
    副本"而非"是否有 stashed base"判定）。
  - K4：merge 段计价增量口径（input + 因果 decode 增长），不随基础
    历史字节膨胀。
  - K5：跨实例 interval gate 重建的 duration µs 下取整（任意 ns 粒度
    不再触发 timer_gate 整 µs 校验崩溃）。
  - K6：deep_gap 台账落账边界——可恢复容量失败不落账，异常携带逐
    rank 缺口；确认终态（守卫）才提交。
  - M1：三态分类按适用动作集合过滤（remote off / PARTIAL 不再被
    remote-read 的 input-only 足迹掩盖为暂时不可行）。
  - M3：CausalHorizonEstimator 版本 + session→run→冷启动优先级
    （覆盖缺口补专项单测）。
  - N1(a) 覆盖缺口：remote-read 适用性收窄单测。
  - 用例 G 对照组：base LOCAL stay vs recompute@home 终态等价。
  - M5：ServiceFactors 样本纯度排除 remote-read decode 列车与
    门控传输首 chunk prefill 列车。
  - 终审-中1/中2：水印 merge 第三方 victim 逐出入账 + joint 重放
    fail-closed 硬化（未知 kind/非整数 bytes/缺实例索引）。
  - 终审-中3：REMOTE 基 + exec==home 的池写 merge 计价。
  - 终审-中4：守卫 raise 内嵌终态缺口记录。
  - 自查 A/B/B'/C/D：守卫事件源补全、metrics 实例过滤、结算释放
    镜像、REMOTE 基池写口径、增长逐出进图+流登记。
  - 四审-低2/低3：自身池写不误判锁死断言、侧车 dump 函数固化。

运行：cd sh_test_mesh/workload/llama2_7b_inference &&
      python3 joint/test_joint_review2_fixes.py   （或 pytest 同路径）
"""
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

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
from joint.test_joint_fixes import (  # noqa: E402
    _manager,
    _make_partial,
    _model,
    _load,
)
from joint.test_joint_mechanisms import (  # noqa: E402
    _seed,
    _tiny_hardware,
    _tiny_model,
    _two_instance_topology,
)
from joint.joint_cost_model import (  # noqa: E402
    ACTION_REMOTE,
    CausalHorizonEstimator,
    JointCostModel,
    LinkFlowRegistry,
    ServiceFactors,
    SessionKVView,
    RequestView,
    _transfer_ns,
)


def _runtime(request_id, session_id=None, joint_action="stay"):
    from online.sh30_online_scheduler import _OnlineRequestRuntime
    runtime = _OnlineRequestRuntime({
        "request_id": request_id,
        "session_id": session_id or ("%s_session" % request_id),
        "turn_index": 0, "queue_index": 0,
        "prefill_length": 512, "decode_length": 8,
        "history_tokens_before": 0,
        "prefill_context_tokens": 512,
        "final_context_tokens": 520,
    }, 512)
    runtime.joint_action = joint_action
    runtime.decode_tokens_consumed = 0
    runtime.prefill_tokens_completed = 0
    runtime.joint_span_base_context = 0
    runtime.remaining_chunks = 0
    return runtime


# ================================================================ K1 守卫 ==

class DeadlockGuardPendingReadyTest(unittest.TestCase):
    """K1：全停滞 + 无 qp/在飞列车，但 pending_decode_ready 非空 → 守卫
    不触发（该队列是活跃事件源：本 pass 列车规划阶段即发射 decode
    列车 → in_flight_train）。"""

    def _scheduler(self):
        from online.sh30_online_scheduler import (
            Sh30OnlineScheduler, _OnlineInstanceState,
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
        # 自查 A 后守卫引用的跨 tick 事件源状态（空 = 无未来事件源）。
        scheduler._pending_merge_alarms = {}
        scheduler.arrival_heap = []
        scheduler.runtimes = []
        scheduler._arrived_request_count = 0
        return scheduler

    def test_pending_decode_ready_escapes_guard(self):
        scheduler = self._scheduler()
        runtime = _runtime("r0", session_id="s0")
        scheduler._enter_decode_stall(runtime, 0, "insufficient HBM")
        scheduler.instances[0].active_decode.append(runtime)
        # 唯一未停滞请求刚 drain（在另一实例待加入列车）。
        scheduler.instances[1].pending_decode_ready.append(_runtime("r1"))
        scheduler._check_decode_deadlock()  # 不得 raise
        # 队列清空（已加入列车并完成）→ 无事件源 → 守卫触发。
        scheduler.instances[1].pending_decode_ready.clear()
        with self.assertRaises(RuntimeError):
            scheduler._check_decode_deadlock()

    def test_pending_merge_alarm_escapes_guard(self):
        """自查 A：完成批刚注册 merge 尾 watch（与 _admit_pass 同 tick
        先后执行）→ watch 必有交付 → 事件源在途，守卫不得触发。"""
        scheduler = self._scheduler()
        runtime = _runtime("r0", session_id="s0")
        scheduler._enter_decode_stall(runtime, 0, "insufficient HBM")
        scheduler.instances[0].active_decode.append(runtime)
        scheduler._pending_merge_alarms = {"batch_train_merge_rq": object()}
        scheduler._check_decode_deadlock()  # 不得 raise
        scheduler._pending_merge_alarms = {}
        with self.assertRaises(RuntimeError):
            scheduler._check_decode_deadlock()

    def test_future_arrival_escapes_guard(self):
        """自查 A：manifest 内尚有未到达请求（C++ 侧已排 alarm）→
        未来到达 → 新准入 → 逐出释放，守卫不得触发；全部到达后才可
        在终态全停滞触发。"""
        scheduler = self._scheduler()
        runtime = _runtime("r0", session_id="s0")
        scheduler._enter_decode_stall(runtime, 0, "insufficient HBM")
        scheduler.instances[0].active_decode.append(runtime)
        scheduler.runtimes = [runtime, _runtime("r1", session_id="s1")]
        scheduler._arrived_request_count = 1  # 2 见 1 → 未来到达存在
        scheduler._check_decode_deadlock()  # 不得 raise
        scheduler._arrived_request_count = 2  # 全部已到达 → 终态可触发
        with self.assertRaises(RuntimeError):
            scheduler._check_decode_deadlock()


# ======================================================== K6 台账边界 ==

class DeepGapLedgerBoundaryTest(unittest.TestCase):
    """K6：可恢复容量失败不污染 deep_gap_events；缺口随异常携带，
    确认终态才落账。"""

    def test_recoverable_failure_carries_records_without_ledger(self):
        kv = _manager(capacity_bytes=51_328)
        _seed(kv, "s", 0, 10, 10, "human")
        _seed(kv, "hog", 0, 300, 10, "human")
        kv._sessions["hog"].active = True
        kv.prepare_prefill(
            session_id="s", target_instance_index=0,
            history_tokens=10, trigger_request_id="t1", action="stay")
        with self.assertRaises(KVCapacityError) as ctx:
            kv.expand_prefill(
                session_id="s", instance_index=0, context_tokens=300,
                trigger_request_id="t1")
        exc = ctx.exception
        self.assertEqual(kv.deep_gap_events, [],
                         "可恢复容量失败不得落 deep_gap 台账")
        self.assertTrue(exc.deep_gap_records,
                        "缺口记录必须随异常携带（逐 rank 明细）")
        self.assertEqual(
            exc.deep_gap_records[0]["trigger_request_id"], "t1")
        # 确认终态 → 提交落账（守卫同款提交点）。
        kv.commit_deep_gap_records(exc.deep_gap_records)
        self.assertEqual(len(kv.deep_gap_events), kv.tp_degree)

    def test_recoverable_merge_degrade_leaves_ledger_empty(self):
        """R4 自降级成功（可恢复路径）不落账（用例 B 口径回归）。"""
        kv = _manager(capacity_bytes=51_328)
        _seed(kv, "s", 0, 10, 10, "human")
        _seed(kv, "hog", 0, 300, 10, "human")
        kv._sessions["hog"].active = True
        kv.prepare_prefill(
            session_id="s", target_instance_index=1,
            history_tokens=10, trigger_request_id="t1", action="copy")
        kv.expand_prefill(
            session_id="s", instance_index=1, context_tokens=14,
            trigger_request_id="t1")
        kv.merge_back(session_id="s", trigger_request_id="t1", new_tokens=4)
        self.assertTrue(kv.merge_degrade_events)
        self.assertEqual(kv.deep_gap_events, [])


# ======================================================== K4 merge 计价 ==

def _session_view(home=0, resident=0, history_bytes=(1000, 1000),
                  location="local_hbm"):
    return SessionKVView(
        session_id="s", home_instance=home, resident_instance=resident,
        location=location, history_tokens=50, resident_prefix_layers=4,
        history_bytes_by_tp_rank=history_bytes,
        missing_bytes_by_tp_rank=(0, 0))


def _request_view(input_tokens=5, decode=10, input_bytes=(500, 500)):
    return RequestView(
        request_id="r", session_id="s",
        input_tokens=input_tokens, history_tokens_before=50,
        estimated_decode_tokens=decode, horizon_source="run_online_mean",
        input_kv_bytes_by_tp_rank=input_bytes)


class MergeIncrementPricingTest(unittest.TestCase):
    """K4：merge 段 = 增量（input + decode 增长）口径。"""

    def test_merge_ns_equals_increment_transfer(self):
        model = _model({0: _load(), 1: _load()})
        session = _session_view()          # home 0，执行 1（异地）
        request = _request_view()          # input 500/rank，decode 增长 1000/rank
        candidate = model.estimate_action(
            session=session, request=request, instance_index=1,
            action="copy", remote_enabled=True)
        self.assertTrue(candidate.applicable)
        # 增量/rank = 500 + 500*10//5 = 1500；总 3000；hops=1、除数 1。
        expected = _transfer_ns(
            total_bytes=3000, path_hops=1, divisor=1, rates=model.rates,
            per_hop_latency_ns=None, startup_ns=0)
        self.assertEqual(candidate.breakdown.merge_ns, expected)

    def test_merge_ns_independent_of_history_bytes(self):
        """基础历史翻倍不改变 merge 段（旧口径会虚增整份工作副本）。"""
        loads = {0: _load(), 1: _load()}
        small = _model(loads).estimate_action(
            session=_session_view(history_bytes=(1000, 1000)),
            request=_request_view(), instance_index=1,
            action="copy", remote_enabled=True)
        big = _model(loads).estimate_action(
            session=_session_view(history_bytes=(5000, 5000)),
            request=_request_view(), instance_index=1,
            action="copy", remote_enabled=True)
        self.assertEqual(
            small.breakdown.merge_ns, big.breakdown.merge_ns,
            "merge 段不得随基础历史字节膨胀（增量口径）")


# ========================================================== K5 timer 整µs ==

class IntervalGateRebuildFloorTest(unittest.TestCase):
    """K5：重建 gate duration = interval + hbm_wait（任意 ns 粒度）下取整
    到 µs——不再触发 timer_gate 整 µs 校验 raise。"""

    def test_sub_us_duration_rebuilds_without_raise(self):
        from online.graph_batch_builder import (
            GraphBatchBuilder, OnlineTraceBuilder, PendingHistoryGate,
            TransferTagAllocator,
        )
        config = SimpleNamespace(
            npus_count=2,
            inference_groups=[SimpleNamespace(ranks=(0,)),
                              SimpleNamespace(ranks=(1,))],
            request_queue=[
                None,
                SimpleNamespace(inter_request_interval_ns=1500)],
            remote_operand_loads=None,
        )
        builder = GraphBatchBuilder.__new__(GraphBatchBuilder)
        builder.config = config
        builder.builders = {
            rank: OnlineTraceBuilder(rank, remote_operand_loads=None)
            for rank in range(2)}
        builder.group_by_index = dict(enumerate(config.inference_groups))
        builder.tag_allocator = TransferTagAllocator()
        # 源实例 rank0 预置一个合法 gate 节点（整 µs 先建）。
        builder.builders[0].set_context("seed", "prefill", 0)
        source_gate = builder.builders[0].timer_gate("seed_gate", 1000)
        self.assertIsNotNone(source_gate)
        pending_gate = PendingHistoryGate(
            source_instance_index=0, timer_gates=(source_gate,),
            location="local_hbm")
        request_plan = {
            "request_id": "req1", "session_id": "sess", "turn_index": 0,
            "queue_index": 1, "prefill_instance_index": 1,
            "hbm_wait_ns": 0,   # 1500 + 0 = 1500 ns：非整 µs（旧代码必崩）
        }
        rebuilt = builder._rebuild_interval_gate_on_target(
            pending_gate, request_plan)
        self.assertEqual(rebuilt.source_instance_index, 1)
        self.assertEqual(len(rebuilt.timer_gates), 1)
        self.assertIsNotNone(rebuilt.timer_gates[0])


# =========================================================== M1 三态分类 ==

class ClassifierActionFilterTest(unittest.TestCase):
    """M1：三态分类按适用动作集合过滤——remote-read 在 remote-off 或
    PARTIAL 基时不参与判定，结构性不可行不被掩盖。"""

    def _scheduler(self, remote_enabled):
        from online.sh30_online_scheduler import Sh30OnlineScheduler
        scheduler = Sh30OnlineScheduler.__new__(Sh30OnlineScheduler)
        scheduler.kv_manager = _manager(capacity_bytes=100_000)
        scheduler.joint_config = SimpleNamespace(
            remote_enabled=remote_enabled)
        return scheduler

    @staticmethod
    def _session_view(kv, session_id):
        session = kv._sessions[session_id]
        return SessionKVView(
            session_id=session_id,
            home_instance=session.home_instance,
            resident_instance=session.instance_index,
            location={"local_hbm": "local_hbm",
                      "partial_hbm_remote": "partial_hbm_remote",
                      "remote_memory": "remote_memory"}.get(
                          session.location, "none"),
            history_tokens=session.context_tokens,
            resident_prefix_layers=session.resident_prefix_layers,
            history_bytes_by_tp_rank=(1, 1),
            missing_bytes_by_tp_rank=(0, 0))

    def test_remote_off_unmasks_structural_infeasibility(self):
        """整份放不下 + remote off → 结构性不可行（旧代码会被 remote-read
        的 input-only 足迹掩盖为暂时不可行）。tiny：128 B/token/rank、
        权重 11,392 B/rank；final 800 需 ≈102 KB/rank > free ≈88 KB。"""
        scheduler = self._scheduler(remote_enabled=False)
        _seed(scheduler.kv_manager, "s", 0, 400, 10, "human")
        runtime = _runtime("r9", session_id="s")
        runtime.joint_input_tokens = 400
        session_view = self._session_view(scheduler.kv_manager, "s")
        structural, detail = (
            scheduler._classify_physical_feasibility(
                runtime, session_view))
        self.assertTrue(structural)
        self.assertFalse(any("remote-read" in line for line in detail))

    def test_remote_on_partial_base_excludes_remote_read(self):
        """PARTIAL 基（后缀不可直读）即使 remote on 也不参与判定。"""
        scheduler = self._scheduler(remote_enabled=True)
        kv = scheduler.kv_manager
        _make_partial(kv, tokens=400)
        runtime = _runtime("r9", session_id="s")
        runtime.joint_input_tokens = 400
        session_view = self._session_view(kv, "s")
        self.assertEqual(session_view.location, "partial_hbm_remote")
        structural, detail = scheduler._classify_physical_feasibility(
            runtime, session_view)
        self.assertTrue(structural)
        self.assertFalse(any("remote-read" in line for line in detail))


# ============================================ M3 时域估计器 + 覆盖缺口 ==

class CausalHorizonEstimatorTest(unittest.TestCase):
    """M3 + 覆盖缺口：session→run→冷启动优先级 + 版本递增。"""

    def test_precedence_and_version(self):
        estimator = CausalHorizonEstimator(cold_start_default_tokens=1)
        self.assertEqual(estimator.version, 0)
        self.assertEqual(estimator.estimate("a"), (1, "cold_start_default"))
        estimator.observe_completed("a", 7)          # run 1 样本
        self.assertEqual(estimator.version, 1)
        self.assertEqual(estimator.estimate("b"), (7, "run_online_mean"))
        estimator.observe_completed("a", 3)          # run 2 样本（a 专属 2 条）
        self.assertEqual(estimator.version, 2)
        self.assertEqual(estimator.estimate("a"), (5, "session_online_mean"))
        # run 均值 = (7+3)/2 = 5：session 无样本的请求回落 run 均值。
        self.assertEqual(estimator.estimate("b"), (5, "run_online_mean"))

    def test_snapshot_cache_invalidates_on_horizon_version(self):
        """估计器更新后，同实例纪元的快照缓存必须失效（M3 核心场景）。"""
        from online.sh30_online_scheduler import (
            Sh30OnlineScheduler, _OnlineInstanceState,
        )
        scheduler = Sh30OnlineScheduler.__new__(Sh30OnlineScheduler)
        scheduler._snapshot_verify = False
        scheduler._joint_horizon = CausalHorizonEstimator(
            cold_start_default_tokens=3)
        state = _OnlineInstanceState(index=0)
        state.snapshot_epoch = state.ledger_epoch
        state.snapshot_cache = object()
        state.snapshot_horizon_version = scheduler._joint_horizon.version
        # 版本一致 → 命中缓存（同一对象）。
        self.assertIs(
            scheduler._task_load_snapshot(state, 0), state.snapshot_cache)
        # 估计器更新（不 bump 实例纪元）→ 缓存必须失效 → 走重算分支。
        scheduler._joint_horizon.observe_completed("s", 5)

        def _fail_recompute(_state):
            raise AssertionError("stale cache was not invalidated")

        scheduler._compute_task_load_snapshot = _fail_recompute
        with self.assertRaises(AssertionError):
            scheduler._task_load_snapshot(state, 0)


class RemoteReadApplicabilityTest(unittest.TestCase):
    """N1(a) 覆盖缺口：remote-read 适用性收窄（LOCAL 基专属）。"""

    def test_remote_read_requires_local_base(self):
        model = _model({0: _load(), 1: _load()})
        request = _request_view()
        for location in ("partial_hbm_remote", "remote_memory"):
            session = _session_view(location=location)
            actions = dict(zip(
                ("stay", "recompute", "copy", "remote-read"),
                model.applicable_actions(
                    session, request, 1, remote_enabled=True)))
            self.assertFalse(actions["remote-read"][0], location)
            self.assertIsNotNone(actions["remote-read"][1])
        local = dict(zip(
            ("stay", "recompute", "copy", "remote-read"),
            model.applicable_actions(
                _session_view(location="local_hbm"), request, 1,
                remote_enabled=True)))
        self.assertTrue(local["remote-read"][0])


class RecomputeAtHomeControlTest(unittest.TestCase):
    """用例 G 对照组：base LOCAL 的 stay 与 recompute@home 终态等价
    （同驻留、同字节、无传输、无工作副本）。"""

    def test_stay_vs_recompute_at_local_home_settle_equal(self):
        results = []
        for action in ("stay", "recompute"):
            kv = _manager()
            _seed(kv, "s", 0, 10, 10, "human")
            _before, transfers, evictions = kv.prepare_prefill(
                session_id="s", target_instance_index=0,
                history_tokens=10, trigger_request_id="t1",
                action=action)
            kv.expand_prefill(
                session_id="s", instance_index=0, context_tokens=14,
                trigger_request_id="t1")
            self.assertEqual(
                kv.merge_back(
                    session_id="s", trigger_request_id="t1", new_tokens=4),
                ())
            snapshot = kv.session_snapshot("s")
            results.append((
                snapshot.location, snapshot.instance_index,
                snapshot.resident_prefix_layers,
                kv.hbm_snapshots(0)[0].kv_cache_bytes))
            kv.mark_complete("s", 20, "human")
        self.assertEqual(results[0], results[1])
        self.assertEqual(results[0][0], KVCacheManager.LOCAL_HBM)
        self.assertEqual(results[0][2], 4)


# ================================================== M5 ServiceFactors 纯度 ==

class ServiceFactorPurityTest(unittest.TestCase):
    """M5：remote-read decode 列车（读流门控）与门控传输首 chunk
    prefill 列车不进样本。"""

    def _scheduler(self):
        from online.sh30_online_scheduler import (
            Sh30OnlineScheduler, _OnlineInstanceState,
        )
        scheduler = Sh30OnlineScheduler.__new__(Sh30OnlineScheduler)
        hardware = _tiny_hardware(10**9)
        scheduler.hardware = hardware
        scheduler.model = _tiny_model(4)
        scheduler.topology = _two_instance_topology(hardware)
        scheduler._prefill_task_cache = {}
        scheduler._decode_task_load_cache = {}
        scheduler._task_load_cache_capacity = 128
        scheduler._joint_factors = ServiceFactors()
        scheduler.runtime_by_request_id = {}
        return scheduler

    def test_remote_read_decode_train_excluded(self):
        scheduler = self._scheduler()
        state = scheduler.instances[0] if hasattr(
            scheduler, "instances") else None
        from online.sh30_online_scheduler import _OnlineInstanceState
        state = _OnlineInstanceState(index=0)
        remote_runtime = _runtime("r0", joint_action="remote-read")
        scheduler.runtime_by_request_id["r0"] = remote_runtime
        train = {
            "emit_tick": 100, "prefill_chunk_tokens": (),
            "members": [("r0", 2)], "had_joiners": False,
        }
        scheduler._observe_service_factors(state, train, 200)
        self.assertEqual(scheduler._joint_factors.updates.get(
            "decode_factor", 0), 0)
        # 同型列车但成员为 stay → 收样（flush 后可见）。
        scheduler.runtime_by_request_id["r0"] = _runtime(
            "r0", joint_action="stay")
        scheduler._observe_service_factors(state, train, 300)
        scheduler._joint_factors.flush()
        self.assertEqual(scheduler._joint_factors.updates.get(
            "decode_factor", 0), 1)

    def test_transfer_gated_prefill_train_excluded(self):
        scheduler = self._scheduler()
        from online.sh30_online_scheduler import _OnlineInstanceState
        state = _OnlineInstanceState(index=0)
        head = _runtime("r1", joint_action="copy")
        scheduler.runtime_by_request_id["r1"] = head
        base_train = {
            "emit_tick": 100, "members": [], "had_joiners": False,
            "suffix_gated": False,
            "prefill_chunk_tokens": (("r1", 512),),
        }
        gated = dict(base_train, history_transfer_gated=True)
        scheduler._observe_service_factors(state, gated, 200)
        self.assertEqual(scheduler._joint_factors.updates.get(
            "prefill_factor", 0), 0)
        pure = dict(base_train, history_transfer_gated=False)
        scheduler._observe_service_factors(state, pure, 300)
        scheduler._joint_factors.flush()
        self.assertEqual(scheduler._joint_factors.updates.get(
            "prefill_factor", 0), 1)


# ================================================== K2/K3 水印真实路径 ==

_SLO_DIR = os.path.join(_WL, "..", "..", "slo_tools")
if _SLO_DIR not in sys.path:
    sys.path.insert(0, _SLO_DIR)
import hbm_watermark  # noqa: E402


def _scan(tokens_requests):
    mapping = hbm_watermark.REPO_VARIANTS["astra-sim-joint"]
    tokens = {"requests": tokens_requests, "path": "test"}
    return hbm_watermark.WatermarkScan(
        Path("/tmp/nonexistent-run"), "astra-sim-joint", mapping, tokens,
        coef=100, capacity=None)


def _token_row(history, prefill_context, final_context, session_id="sess"):
    return {
        "session_id": session_id,
        "history_tokens_before": history,
        "prefill_context_tokens": prefill_context,
        "final_context_tokens": final_context,
    }


class WatermarkCompletionSettlementTest(unittest.TestCase):
    """K2/K3：驱动真实 WatermarkScan.consume（joint 分支），断言完成
    结算的双计回归与 REMOTE 工作副本泄漏回归。coef=100 B/token。"""

    def test_cross_instance_completion_settles_increment_only(self):
        scan = _scan({
            "q1": _token_row(0, 10, 10),
            "q2": _token_row(10, 14, 16),
        })
        # turn-1（stay@0）：驻留 10 token → home 1000 B。
        scan.consume({"request_id": "q1", "kind": "prefill", "tick": 1,
                      "decision": {"joint_action": "stay",
                                   "prefill_instance_index": 0}})
        scan.consume({"request_id": "q1", "kind": "decode", "tick": 2,
                      "decision": {"joint_action": "stay",
                                   "decode_instance_index": 0}})
        scan.consume({"request_id": "q1", "kind": "completion", "tick": 3,
                      "decision": {"joint_action": "stay"}})
        self.assertEqual(scan.replay.occupancy, {0: 1000})
        # turn-2（copy@1）：stash base；noc 前缀 1000 B → 实例 1。
        scan.consume({"request_id": "q2", "kind": "prefill", "tick": 4,
                      "decision": {
                          "joint_action": "copy",
                          "prefill_instance_index": 1,
                          "origin_home_instance": 0,
                          "history_transfers": [
                              {"kind": "noc_migrate",
                               "total_bytes": 1000}]}})
        # noc 1000 + 增长到 prefill 上下文 14 token（1400 B）。
        self.assertEqual(scan.replay.occupancy, {0: 1000, 1: 1400})
        scan.consume({"request_id": "q2", "kind": "decode", "tick": 5,
                      "decision": {"joint_action": "copy",
                                   "decode_instance_index": 1}})
        # 完成时执行端已增长到 16 token（1600 B）。
        self.assertEqual(scan.replay.occupancy, {0: 1000, 1: 1600})
        scan.consume({"request_id": "q2", "kind": "completion", "tick": 6,
                      "decision": {
                          "joint_action": "copy",
                          "origin_home_instance": 0,
                          "joint_working_copy": True,
                          "merge_transfers": [
                              {"kind": "noc_migrate",
                               "total_bytes": 200}]}})
        # K2 核心：home = base(1000) + 增量(200) = 1200——旧代码
        # 双计 base → 2200；执行端释放归零。
        self.assertEqual(scan.replay.occupancy, {0: 1200, 1: 0})
        session = scan.replay.sessions["sess"]
        self.assertEqual((session.instance, session.bytes), (0, 1200))
        self.assertIsNone(session.base_instance)

    def test_remote_base_working_copy_released(self):
        scan = _scan({
            "q1": _token_row(0, 12, 12),
        })
        # REMOTE 基（无 home 驻留）：copy 从池恢复整份到实例 1。
        scan.consume({"request_id": "q1", "kind": "prefill", "tick": 1,
                      "decision": {
                          "joint_action": "copy",
                          "prefill_instance_index": 1,
                          "history_transfers": [
                              {"kind": "remote_load",
                               "total_bytes": 1000}]}})
        scan.consume({"request_id": "q1", "kind": "decode", "tick": 2,
                      "decision": {"joint_action": "copy",
                                   "decode_instance_index": 1}})
        self.assertEqual(scan.replay.occupancy, {1: 1200})
        # K3 核心：REMOTE 基无 stashed base，但工作副本必须释放
        # （旧代码早退 → 永久残留）。
        scan.consume({"request_id": "q1", "kind": "completion", "tick": 3,
                      "decision": {
                          "joint_action": "copy",
                          "origin_home_instance": 0,
                          "joint_working_copy": True,
                          "merge_transfers": [
                              {"kind": "remote_store", "reason":
                               "merge_increment_suffix_pool_store",
                               "session_id": "sess",
                               "total_bytes": 400}]}})
        self.assertEqual(scan.replay.occupancy, {1: 0})
        session = scan.replay.sessions["sess"]
        self.assertIsNone(session.instance)
        self.assertEqual(session.bytes, 0)
        # 四审-低2 锁死断言：自身增量池写不得被判第三方逐出——工作副本
        # 释放归零使占用断言对误判不敏感（回归保护缺口），evictions
        # 计数器是敏感判别器（误判必使 apply_evict 计数 > 0）。
        self.assertEqual(
            scan.replay.report.actions.get("evictions", 0), 0)


# ==================================================== 自查 B/B' metrics ==

class MetricsInstanceFilterTest(unittest.TestCase):
    """自查 B（2026-09-15）：跨实例会话的 metrics parts 按实例截断
    ——自降级只动 home 端；执行端工作副本释放只动 exec 端。"""

    def test_suffix_evict_parts_filters_by_instance(self):
        import face_scheduler as fs
        kv = _manager()
        # 同会话 parts 落两实例：home 基础（0..2 层）+ exec 工作副本。
        fs.set_metrics_observer(_RecordingRecorder())
        try:
            kv2 = _manager()
            _seed(kv2, "s", 0, 10, 10, "human")
            kv2.prepare_prefill(
                session_id="s", target_instance_index=1,
                history_tokens=10, trigger_request_id="t1", action="copy")
            kv2.expand_prefill(
                session_id="s", instance_index=1, context_tokens=14,
                trigger_request_id="t1")
            parts = kv2._metrics_parts["s"]
            instances = {part["instance_index"] for part in parts}
            self.assertEqual(instances, {0, 1})
            # 只截 home（实例 0）的 parts：exec parts 逐字节不动。
            exec_before = [dict(part) for part in parts
                           if part["instance_index"] == 1]
            kv2._metrics_suffix_evict_parts(
                "s", 1, anchor_kind="completion", request_id="t1",
                cause="probe", instance_index=0)
            exec_after = [dict(part) for part in kv2._metrics_parts["s"]
                          if part["instance_index"] == 1]
            self.assertEqual(exec_before, exec_after)
            home = [part for part in kv2._metrics_parts["s"]
                    if part["instance_index"] == 0]
            self.assertTrue(all(part["layer_end"] <= 1 for part in home))
        finally:
            fs.set_metrics_observer(None)

    def test_merge_release_clears_exec_parts(self):
        """自查 B'：merge_back 结算后该会话在 exec 实例的 parts 必须
        全量移除（旧代码漏镜像 → metrics 通道 exec 永久高估）。"""
        import face_scheduler as fs
        fs.set_metrics_observer(_RecordingRecorder())
        try:
            kv = _manager()
            _seed(kv, "s", 0, 10, 10, "human")
            kv.prepare_prefill(
                session_id="s", target_instance_index=1,
                history_tokens=10, trigger_request_id="t1", action="copy")
            kv.expand_prefill(
                session_id="s", instance_index=1, context_tokens=14,
                trigger_request_id="t1")
            kv.merge_back(session_id="s", trigger_request_id="t1",
                          new_tokens=4)
            exec_parts = [part for part in kv._metrics_parts.get("s", [])
                          if part["instance_index"] == 1]
            self.assertEqual(exec_parts, [])
        finally:
            fs.set_metrics_observer(None)


class _RecordingRecorder:
    """metrics recorder 最小桩：只接受 initialize_rank/record 调用
    （本组断言走 _metrics_parts 结构，不依赖行内容）。"""

    def initialize_rank(self, rank, capacity_bytes):
        return None

    def record(self, **kwargs):
        return None


# ================================================== 自查 C REMOTE 基计价 ==

class RemoteBaseMergePricingTest(unittest.TestCase):
    """自查 C：REMOTE 基会话的 merge 段按池写口径计价（执行端池端口
    除数、无 home 空间准备），不再误用 NoC→home 路由。"""

    def test_remote_base_merge_uses_pool_caliber(self):
        model = _model({0: _load(), 1: _load()})
        session = SessionKVView(
            session_id="s", home_instance=0, resident_instance=None,
            location="remote_memory", history_tokens=50,
            resident_prefix_layers=0,
            history_bytes_by_tp_rank=(0, 0),
            missing_bytes_by_tp_rank=(5000, 5000))
        request = _request_view()
        candidate = model.estimate_action(
            session=session, request=request, instance_index=1,
            action="copy", remote_enabled=True)
        self.assertTrue(candidate.applicable)
        from joint.joint_cost_model import _pool_transfer_ns
        expected = _pool_transfer_ns(
            total_bytes=3000, divisor=model._pool_divisor(1),
            rates=model.rates)
        self.assertEqual(candidate.breakdown.merge_ns, expected)
        self.assertIn("merge_to_pool_backing",
                      candidate.breakdown.notes)
        self.assertNotIn("merge_to_home=0", candidate.breakdown.notes)

    def test_local_base_still_uses_noc_caliber(self):
        """LOCAL 基（跨实例）保持 NoC→home + home 空间准备口径。"""
        model = _model({0: _load(), 1: _load()})
        candidate = model.estimate_action(
            session=_session_view(), request=_request_view(),
            instance_index=1, action="copy", remote_enabled=True)
        self.assertIn("merge_to_home=0", candidate.breakdown.notes)
        self.assertNotIn("merge_to_pool_backing",
                         candidate.breakdown.notes)


# ================================================== 自查 D 增长逐出进图 ==

class GrowthEvictionEmissionTest(unittest.TestCase):
    """自查 D：逐列车 decode 增长的已提交逐出必须进图（旁路支链）
    + R15 流登记（rid#decode owner）——三路（成功/停滞/唤醒）同款。"""

    def _scheduler(self):
        from online.sh30_online_scheduler import (
            Sh30OnlineScheduler, _OnlineInstanceState,
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
        scheduler._pending_merge_alarms = {}
        scheduler.arrival_heap = []
        scheduler.runtimes = []
        scheduler._arrived_request_count = 0
        scheduler.graph = _RecordingGraph()
        from joint.joint_cost_model import LinkFlowRegistry
        scheduler._joint_flows = LinkFlowRegistry()
        scheduler._pool_ports = _RecordingPoolPorts()
        scheduler._instance_edge_ports = {}
        return scheduler

    def test_growth_evictions_emitted_and_flow_registered(self):
        scheduler = self._scheduler()
        kv = _manager(capacity_bytes=51_328)  # 紧容量：增长必触发逐出
        scheduler.kv_manager = kv
        # 容量算术（128 B/token/rank、权重 11,392 B/rank）：
        # hog=300(38,400) + victim=10(1,280) + s=2(256) + 权重
        # = 51,328 恰满（free=0）；增长 +2 token(256B/rank) 必逐 victim。
        # 紧容量下 s 必须最后种入（先种的已完成会话是合法逐出 victim）。
        _seed(kv, "hog", 0, 300, 10, "human")
        kv._sessions["hog"].active = True
        _seed(kv, "victim", 0, 10, 10, "human")
        _seed(kv, "s", 0, 2, 10, "human")
        runtime = _runtime("r0", session_id="s")
        runtime.history_tokens_before = 0
        runtime.joint_input_tokens = 2
        runtime.decode_tokens_consumed = 5  # 目标上下文 7 > 当前 2：增长
        # 5 token(640B/rank) > effective remaining(≈304，不计可逐会话)
        # → 必逐 victim（completed 会话为合法 victim）。
        scheduler._batch = {"tick": 42}
        scheduler._joint_grow_decode(runtime, 0)
        emitted = scheduler.graph.emitted
        self.assertTrue(emitted, "增长逐出必须进图（旁路支链）")
        # remote_store 的 noc_path 为单节点（无链路可登）——池写争用经
        # 池端口登记（_pool_ports），owner 同为 rid#decode。
        self.assertTrue(scheduler._pool_ports._counts,
                        "增长逐出必须登记池端口份额（除数口径）")
        scheduler._pool_ports.release_owner("r0#decode")
        released = scheduler._joint_flows.release_owner("r0#decode")
        self.assertGreaterEqual(released, 0)  # 链路登记可为空（单节点路径）


class _RecordingGraph:
    """构图器最小桩：记录旁路支链发射。"""

    def __init__(self):
        self.emitted = []

    def sync_pending_history_after_evictions(self, transfers):
        return None

    def emit_eviction_side_branch(self, transfers, tick):
        self.emitted.append((tuple(transfers), tick))


class _RecordingPoolPorts:
    """池端口登记最小桩。"""

    def __init__(self):
        self._counts = {}

    def register(self, edge_rank, *, owner):
        self._counts[edge_rank] = self._counts.get(edge_rank, 0) + 1

    def release_owner(self, owner):
        return 0


# ============================================ 终审-中1/中2/中3（kimi 三审） ==

class WatermarkThirdPartyEvictionTest(unittest.TestCase):
    """终审-中1：merge 空间准备的第三方 victim 逐出入水印重放
    （home_merge_capacity remote_store，victim=其它会话）。"""

    def test_merge_capacity_victim_eviction_accounted(self):
        scan = _scan({
            "qv": _token_row(0, 10, 10, session_id="victim"),
            "q1": _token_row(0, 10, 10),
            "q2": _token_row(10, 14, 16),
        })
        # victim 与 sess turn-1 都驻实例 0（各 1000B）。
        scan.consume({"request_id": "qv", "kind": "prefill", "tick": 1,
                      "decision": {"joint_action": "stay",
                                   "prefill_instance_index": 0}})
        scan.consume({"request_id": "q1", "kind": "prefill", "tick": 2,
                      "decision": {"joint_action": "stay",
                                   "prefill_instance_index": 0}})
        self.assertEqual(scan.replay.occupancy, {0: 2000})
        # sess turn-2 copy@1：stash base；noc 前缀 1000B → 实例 1。
        scan.consume({"request_id": "q2", "kind": "prefill", "tick": 3,
                      "decision": {
                          "joint_action": "copy",
                          "prefill_instance_index": 1,
                          "origin_home_instance": 0,
                          "history_transfers": [
                              {"kind": "noc_migrate",
                               "total_bytes": 1000}]}})
        scan.consume({"request_id": "q2", "kind": "decode", "tick": 4,
                      "decision": {"joint_action": "copy",
                                   "decode_instance_index": 1}})
        self.assertEqual(scan.replay.occupancy, {0: 2000, 1: 1600})
        # 完成：增量 noc 200 回 home + **victim 300B 被 home 侧空间
        # 准备逐出**（remote_store/home_merge_capacity）+ 工作副本释放。
        scan.consume({"request_id": "q2", "kind": "completion", "tick": 5,
                      "decision": {
                          "joint_action": "copy",
                          "origin_home_instance": 0,
                          "joint_working_copy": True,
                          "merge_transfers": [
                              {"kind": "noc_migrate",
                               "total_bytes": 200},
                              {"kind": "remote_store",
                               "reason": "home_merge_capacity",
                               "session_id": "victim",
                               "total_bytes": 300}]}})
        # home = victim(1000-300) + sess base(1000) + 增量(200) = 1900。
        self.assertEqual(scan.replay.occupancy, {0: 1900, 1: 0})
        victim = scan.replay.sessions["victim"]
        self.assertEqual((victim.instance, victim.bytes), (0, 700))


class WatermarkJointHardeningTest(unittest.TestCase):
    """终审-中2：joint 重放容错硬化——未知 kind / 非整数 bytes /
    缺实例索引均 fail-closed（与通用路径同级）。"""

    def _base_rows(self, scan):
        scan.consume({"request_id": "q1", "kind": "prefill", "tick": 1,
                      "decision": {"joint_action": "stay",
                                   "prefill_instance_index": 0}})
        scan.consume({"request_id": "q1", "kind": "decode", "tick": 2,
                      "decision": {"joint_action": "stay",
                                   "decode_instance_index": 0}})

    def test_unknown_merge_kind_fails(self):
        import hbm_watermark
        scan = _scan({"q1": _token_row(0, 10, 10)})
        self._base_rows(scan)
        with self.assertRaises(hbm_watermark.SloToolError):
            scan.consume({"request_id": "q1", "kind": "completion",
                          "tick": 3,
                          "decision": {
                              "joint_action": "stay",
                              "merge_transfers": [
                                  {"kind": "teleport",
                                   "total_bytes": 100}]}})

    def test_non_integer_bytes_fails(self):
        import hbm_watermark
        scan = _scan({"q1": _token_row(0, 10, 10)})
        with self.assertRaises(hbm_watermark.SloToolError):
            scan.consume({"request_id": "q1", "kind": "prefill", "tick": 1,
                          "decision": {
                              "joint_action": "copy",
                              "prefill_instance_index": 1,
                              "history_transfers": [
                                  {"kind": "noc_migrate",
                                   "total_bytes": 100.5}]}})

    def test_missing_prefill_instance_fails(self):
        import hbm_watermark
        scan = _scan({"q1": _token_row(0, 10, 10)})
        with self.assertRaises(hbm_watermark.SloToolError):
            scan.consume({"request_id": "q1", "kind": "prefill", "tick": 1,
                          "decision": {"joint_action": "copy"}})


class RemoteBaseAtHomeMergePricingTest(unittest.TestCase):
    """终审-中3：REMOTE 基 + exec==home（copy@home 退化池恢复）——
    执行侧 merge_back 恒走池写，计价不得因 home==exec 免单。"""

    def test_remote_base_at_home_charges_pool_merge(self):
        model = _model({0: _load(), 1: _load()})
        session = SessionKVView(
            session_id="s", home_instance=0, resident_instance=None,
            location="remote_memory", history_tokens=50,
            resident_prefix_layers=0,
            history_bytes_by_tp_rank=(0, 0),
            missing_bytes_by_tp_rank=(5000, 5000))
        candidate = model.estimate_action(
            session=session, request=_request_view(),
            instance_index=0, action="copy", remote_enabled=True)
        self.assertTrue(candidate.applicable)
        self.assertGreater(candidate.breakdown.merge_ns, 0,
                           "REMOTE 基 exec==home 的池写 merge 不得免单")
        self.assertIn("merge_to_pool_backing", candidate.breakdown.notes)


class GuardMessageEmbedsRecordsTest(unittest.TestCase):
    """终审-中4：死锁守卫 raise 消息内嵌全部终态缺口记录。"""

    def test_raise_message_contains_records(self):
        from online.sh30_online_scheduler import (
            Sh30OnlineScheduler, _OnlineInstanceState,
        )
        scheduler = Sh30OnlineScheduler.__new__(Sh30OnlineScheduler)
        scheduler.instances = [_OnlineInstanceState(index=0)]
        scheduler.kv_manager = _manager()
        scheduler._kv_ledger_epoch = 0
        scheduler._stalled_by_instance = {}
        scheduler._batch = None
        scheduler.online_log_count = 0
        scheduler.decision_log_sink = None
        scheduler.online_log_rows = []
        scheduler._pending_merge_alarms = {}
        scheduler.arrival_heap = []
        scheduler.runtimes = []
        scheduler._arrived_request_count = 0
        runtime = _runtime("r0", session_id="s0")
        records = ({"instance_index": 0, "rank": 0, "phase": "decode",
                    "reason": "grow", "trigger_request_id": "r0",
                    "gap_bytes": 12345},)
        scheduler._enter_decode_stall(
            runtime, 0, "insufficient HBM", gap_records=records)
        scheduler.instances[0].active_decode.append(runtime)
        with self.assertRaises(RuntimeError) as ctx:
            scheduler._check_decode_deadlock()
        self.assertIn("deep_gap_records=", str(ctx.exception))
        self.assertIn("12345", str(ctx.exception))
        # 终态提交后台账含该记录（消息与侧车同源）。
        self.assertEqual(len(scheduler.kv_manager.deep_gap_events), 1)


# ==================================================== 四审低项固化 ==

class SidecarDumpFunctionTest(unittest.TestCase):
    """四审-低3：dump_joint_kv_ledgers 模块级函数固化（终审-中4 的
    finally 落盘调用点）+ 原子写（低8）。"""

    def test_dump_writes_both_ledgers_atomically(self):
        import tempfile
        from online.online_service import dump_joint_kv_ledgers
        kv = _manager()
        kv.merge_degrade_events.append({"session_id": "s", "probe": 1})
        kv.deep_gap_events.append({"rank": 0, "probe": 2})
        scheduler = SimpleNamespace(kv_manager=kv)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "joint_kv_ledgers.json")
            dump_joint_kv_ledgers(scheduler, path)
            import json
            data = json.load(open(path))
            self.assertEqual(data["merge_degrade_events"],
                             [{"probe": 1, "session_id": "s"}])
            self.assertEqual(data["deep_gap_events"], [{"probe": 2, "rank": 0}])
            self.assertFalse(os.path.exists(path + ".tmp"))
            # 落盘失败不抛（诊断通道不得阻断主路径退出语义）。
            dump_joint_kv_ledgers(scheduler, os.path.join(tmp, "no", "dir"))
        # RED 语义快查：守卫提交后的终态记录经同一函数可完整落盘。
        kv.deep_gap_events.append({"rank": 1, "gap_bytes": 9})
        with tempfile.TemporaryDirectory() as tmp:
            dump_joint_kv_ledgers(
                scheduler, os.path.join(tmp, "again.json"))


class MissingBytesFailsTest(unittest.TestCase):
    """四审-低7：total_bytes 缺失（None）同样 raise，不静默归零。"""

    def test_missing_total_bytes_fails(self):
        import hbm_watermark
        scan = _scan({"q1": _token_row(0, 10, 10)})
        with self.assertRaises(hbm_watermark.SloToolError):
            scan.consume({"request_id": "q1", "kind": "prefill", "tick": 1,
                          "decision": {
                              "joint_action": "copy",
                              "prefill_instance_index": 1,
                              "history_transfers": [
                                  {"kind": "noc_migrate"}]}})


if __name__ == "__main__":
    unittest.main()
