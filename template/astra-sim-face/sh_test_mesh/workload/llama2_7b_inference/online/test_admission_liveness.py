#!/usr/bin/env python3
"""test_admission_liveness.py -- P0-1 准入重试自驱动 + P1 选择期容量可行性
过滤测试(2026-08-31;总文档 §4 P0-1/P1,执行文档批次1)。

覆盖(执行文档 §2.7):
  用例 A 单阻塞恢复:X 的 decode target 被在飞 Y 占用 -> 阻塞即置
    decode_admission_dirty(P0-1 契约);Y 完成的下一决策边界 X 被准入。
    旧语义(无逐出阻塞不推进 capacity_epoch)下纪元门已同步关闭,
    dirty 是唯一重开通道——重放 2026-08-31 全量 tracelab window=0 死端
    (306 悬置)的最小内核形态。
  用例 C prefill 卡死重试:prepare_history 阻塞(无逐出)即置
    prefill_admission_dirty;不变状态下再次调用确实重新尝试(旧纪元门
    会直接 return False);容量释放后准入成功。
  P1 decode 三态:原选可行->保持且零日志(决策零扰动);原选不可行
    ->可行集内按 per_die_delta_ns 重选;全不可行->保持原选(交给
    P0-1 阻塞-重试路径)。
  P1 prefill 三态:同上(ordering_key 口径)。
  零副作用:预检只读——hbm/session 快照、capacity_epoch、KV 事件流
    在失败探测前后逐项不变(总文档 §5 不变量 5)。

2026-10-20 方案一(FACE 仓)追加:
  信号触点回归锁:审计表认定的容量释放路径(逐出完成/retirement/
    workspace 外部视图回落/drain 准入)逐一锁定 capacity_epoch 推进或
    dirty 重试门信号(风险一活性加固,审计结论"无缺口"的回归锁);
  阻塞行增富:admission_blocked 决策行的 stage/instance_index/
    capacity_epoch/kv_free_bytes_by_rank/required_bytes_by_rank 数值
    断言(LOCAL_HBM 同实例增量口径 + 无会话全量口径两形态);
  streak 簿记与 verify_run_end 终检:连续阻塞计数/进展清账/直方图,
    run 结束仍阻塞即 raise;
  风险三地板对账:workspace 峰值越 prospective 证书地板即 raise,
    峰值内产出逐 rank slack 字典;
  离线活性审计器四态单测(online/verify/admission_liveness_audit.py):
    PASS(无阻塞行)/PASS(短 streak)/FAIL(>=3 连续可行 streak)/
    SKIP(增富字段缺失的长 streak)+ run 目录自动定位。

测试组织说明(与 online/ 既有测试同风格):被测方法全部为真实实现;
协作者(kv_manager/instances/runtimes)为真实对象;仅容器经
FaceOnlineScheduler.__new__ 手工装配聚焦属性(完整构造的集成路径由
runner 冒烟覆盖,见执行文档批次3)。KV 状态一律经与真实准入路径相同的
prepare_history/grow_prefill/mark_complete 调用建立。

运行:cd sh_test_mesh/workload/llama2_7b_inference &&
      python3 online/test_admission_liveness.py   （或 pytest 同路径）
"""
import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from collections import deque
from pathlib import Path
from types import SimpleNamespace

_ONLINE_DIR = os.path.dirname(os.path.abspath(__file__))
_WORKLOAD_DIR = os.path.dirname(_ONLINE_DIR)
for _p in (_ONLINE_DIR, _WORKLOAD_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import online.verify.admission_liveness_audit as admission_liveness_audit  # noqa: E402
from face_scheduler import (  # noqa: E402
    DecodeTieCounter,
    FaceHardware,
    FaceInstanceSpec,
    WeightedInstanceGraph,
    build_instances,
)
import online.face_online_scheduler as face_online_scheduler  # noqa: E402
from online.face_online_scheduler import (  # noqa: E402
    FaceOnlineScheduler,
    _OnlineInstanceState,
    _OnlineRequestRuntime,
)
from session_kv_manager import (  # noqa: E402
    RESIDENT,
    EvictionRecord,
    SessionKVCacheManager,
    kv_cache_shard_bytes_for_tokens,
)

# 合成小模型(tp=2,8 heads 均分 4/4 -> 每 rank KV 字节/token 相等,
# 算术可由 kv_cache_shard_bytes_for_tokens 在测试内权威复算)。
MODEL = SimpleNamespace(
    layers=2, hidden_size=64, num_heads=8, bytes_per_elem=2,
    ffn_size=128, vocab_size=256, mlp_variant="gelu",
)
TP = 2
CAPACITY_BYTES = 1_000_000
P_CHUNK = 128


def _hardware():
    return FaceHardware(
        mesh_rows=3, mesh_cols=2,
        local_hbm_capacity_bytes=CAPACITY_BYTES,
        local_hbm_bandwidth_gbps=1.0, d2d_bandwidth_gbps=2.0,
        peak_perf_tflops=1.0, d2d_latency_ns=0, local_hbm_latency_ns=0,
    )


def _topology():
    return build_instances(
        _hardware(),
        (
            FaceInstanceSpec("ins0", "1", (0, 1)),
            FaceInstanceSpec("ins1", "2", (2, 3)),
            FaceInstanceSpec("ins2", "3", (4, 5)),
        ),
    )


def _shard_per_rank(tokens: int) -> int:
    shards = kv_cache_shard_bytes_for_tokens(MODEL, tokens, TP)
    assert len(set(shards)) == 1, "合成模型应每 rank 等分"
    return shards[0]


def _tokens_for_bytes(target_bytes: int) -> int:
    """每 rank 字节需求 -> token 数(向上取整,保守够量)。"""
    per_token = _shard_per_rank(1)
    return -(-target_bytes // per_token)


def _make_scheduler():
    hardware = _hardware()
    topology = _topology()
    scheduler = FaceOnlineScheduler.__new__(FaceOnlineScheduler)
    scheduler.topology = topology
    scheduler.config = SimpleNamespace(model=MODEL, hardware=hardware)
    scheduler.kv_manager = SessionKVCacheManager(topology, MODEL)
    scheduler.instances = [
        _OnlineInstanceState(index=i) for i in range(len(topology.instances))]
    scheduler.waiting_decode_admissions = {
        i: deque() for i in range(len(topology.instances))}
    scheduler.capacity_epoch = [0] * len(topology.instances)
    scheduler.decode_admission_epoch = [-1] * len(topology.instances)
    scheduler.decode_admission_dirty = set()
    scheduler.prefill_admission_dirty = set()
    scheduler._ready_frontier = set()
    scheduler._batch = {"assignments": []}
    scheduler.online_log_count = 0
    scheduler.decision_log_sink = None
    scheduler.online_log_rows = []
    scheduler.ledger_admitted = {}
    scheduler.p_chunk = P_CHUNK
    face_online_scheduler._P_CHUNK_HOLDER[0] = P_CHUNK
    # 风险一/风险三观测簿记(2026-10-20 方案一;镜像调度器 __init__ 的
    # 纯观测属性,__new__ 聚焦装配由夹具补齐)。
    scheduler._admission_blocked_streak = {}
    scheduler._admission_blocked_streak_max = {}
    scheduler._admission_blocked_open = {}
    scheduler._workspace_peak_bytes_by_rank = {}
    scheduler.admission_blocked_streak_histogram = None
    scheduler.workspace_floor_reconciliation = None
    scheduler.workspace_admission = "off"
    return scheduler


def _make_runtime(request_id, session_id, *, turn=0, queue_index=0,
                  prefill_context_tokens, final_context_tokens=None,
                  prefill_length=None, decode_length=1,
                  history_tokens_before=0):
    return _OnlineRequestRuntime({
        "request_id": request_id,
        "session_id": session_id,
        "turn_index": turn,
        "queue_index": queue_index,
        "prefill_length": (
            prefill_length if prefill_length is not None
            else prefill_context_tokens),
        "decode_length": decode_length,
        "history_tokens_before": history_tokens_before,
        "prefill_context_tokens": prefill_context_tokens,
        "final_context_tokens": (
            final_context_tokens if final_context_tokens is not None
            else prefill_context_tokens),
    })


def _kv_admit_prefill(manager, session_id, instance_index, tokens,
                      now_ns, request_id):
    """经与真实准入相同的调用序列把会话置为 RESIDENT+active。"""
    decision = manager.prepare_history(
        session_id, instance_index, 0, now_ns, request_id,
        required_context_tokens=tokens)
    assert not decision.admission_blocked, "fixture 预放置必须可行"
    growth = manager.grow_prefill(session_id, tokens, now_ns, request_id)
    assert growth.admitted, "fixture 预放置 grow_prefill 必须成功"


def _occupancy_tokens(manager):
    """空实例每 rank 可用 KV 字节(扣模型权重)。"""
    snapshot = manager.hbm_snapshots(0)
    return min(state.remaining_bytes for state in snapshot)


def _pressure_events(manager):
    return [
        event for event in manager.events_since(0)
        if event.event_type in ("admission_blocked", "admission_retry")
    ]


class _PChunkHolderGuard:
    """_make_scheduler() 会改写生产模块全局 _P_CHUNK_HOLDER[0](调度器
    __init__ 同款机制,face_online_scheduler.py 的调度器 __init__);此处 setUp 保存、
    tearDown 还原,防止跨测试模块的状态泄漏(生产全局不残留本测试的
    P_CHUNK=128)。"""

    def setUp(self):
        self._p_chunk_before = face_online_scheduler._P_CHUNK_HOLDER[0]

    def tearDown(self):
        face_online_scheduler._P_CHUNK_HOLDER[0] = self._p_chunk_before


class DecodeAdmissionLivenessTest(_PChunkHolderGuard, unittest.TestCase):
    """用例 A:decode 阻塞即重开重试门;blocker 完成的下一边界准入。"""

    def test_blocked_then_admitted_after_blocker_completes(self):
        scheduler = _make_scheduler()
        manager = scheduler.kv_manager
        avail = _occupancy_tokens(manager)
        # X 上下文占空实例 ~60%;Y 占 inst1 剩余 -> X 的全量 handoff 放不下。
        tokens_x = _tokens_for_bytes(int(avail * 0.6))
        tokens_y = _tokens_for_bytes(avail - _shard_per_rank(tokens_x) + 1)
        _kv_admit_prefill(manager, "sx", 0, tokens_x, 100, "rx")
        _kv_admit_prefill(manager, "sy", 1, tokens_y, 100, "ry")
        self.assertGreater(
            _shard_per_rank(tokens_y), avail - _shard_per_rank(tokens_x))

        runtime = _make_runtime(
            "rx", "sx", prefill_context_tokens=tokens_x,
            final_context_tokens=tokens_x)
        runtime.prefill_instance_index = 0
        runtime.decode_instance_index = 1
        runtime.waiting_decode_admission = True
        scheduler.waiting_decode_admissions[1].append(runtime)

        # 第一个决策边界:阻塞(无逐出——sy 是 ACTIVE,不可逐出)。
        scheduler._try_admit_waiting_decodes(1000)
        self.assertTrue(runtime.waiting_decode_admission)
        self.assertEqual(len(scheduler.waiting_decode_admissions[1]), 1)
        # P0-1 契约:阻塞即置 dirty;纪元门本身已关闭(同步于扫描头),
        # dirty 是唯一重开通道(旧实现此处为空集 -> 隔离时间线永久悬置)。
        self.assertIn(1, scheduler.decode_admission_dirty)
        self.assertEqual(scheduler.decode_admission_epoch[1],
                         scheduler.capacity_epoch[1])
        self.assertEqual(scheduler.capacity_epoch[1], 0)

        # 容量完全不变的重试确实发生(旧纪元门会静默跳过):首次重复
        # 阻塞至少产生一条 admission_retry 审计事件。
        events_before = len(_pressure_events(manager))
        scheduler._try_admit_waiting_decodes(1001)
        self.assertTrue(runtime.waiting_decode_admission)
        self.assertGreater(len(_pressure_events(manager)), events_before)

        # blocker Y 完成 -> 剩余容量可逐出让位 -> 下一决策边界 X 准入。
        manager.mark_complete("sy", 2000, "ry")
        scheduler._try_admit_waiting_decodes(2000)
        self.assertFalse(runtime.waiting_decode_admission)
        self.assertEqual(list(scheduler.waiting_decode_admissions[1]), [])
        self.assertIn(runtime, scheduler.instances[1].pending_decode_ready)
        self.assertTrue(any(row["kind"] == "decode"
                            for row in scheduler.online_log_rows))
        self.assertEqual(
            scheduler._batch["assignments"][-1]["decode_instance_index"], 1)


class PrefillAdmissionLivenessTest(_PChunkHolderGuard, unittest.TestCase):
    """用例 C:prefill 阻塞即置 dirty;不变状态重试;释放后准入。"""

    def test_blocked_marks_dirty_and_retries_then_admits(self):
        scheduler = _make_scheduler()
        manager = scheduler.kv_manager
        avail = _occupancy_tokens(manager)
        tokens_x = _tokens_for_bytes(int(avail * 0.6))
        tokens_y = _tokens_for_bytes(avail - _shard_per_rank(tokens_x) + 1)
        _kv_admit_prefill(manager, "sy", 1, tokens_y, 100, "ry")

        runtime = _make_runtime(
            "rx", "sx", prefill_context_tokens=tokens_x)
        runtime.prefill_instance_index = 1

        # 第一次尝试:阻塞(无逐出,ACTIVE 的 sy 不可逐出)。
        self.assertFalse(scheduler._try_admit_prefill(runtime, 3000))
        self.assertFalse(runtime.admitted_prefill)
        self.assertIn(1, scheduler.prefill_admission_dirty)
        self.assertEqual(runtime.prefill_attempt_epoch,
                         scheduler.capacity_epoch[1])

        # 不变状态下再次调用确实重新尝试(旧纪元门直接 return False,
        # 不会重新取 hbm 快照):每次过门尝试都会重新赋值
        # hbm_before_request,以对象身份变化为证。
        snapshot_obj = runtime.hbm_before_request
        self.assertFalse(scheduler._try_admit_prefill(runtime, 3001))
        self.assertIsNot(runtime.hbm_before_request, snapshot_obj)
        self.assertIn(1, scheduler.prefill_admission_dirty)

        # blocker 完成 -> 冷会话可逐出 -> 准入成功。
        manager.mark_complete("sy", 4000, "ry")
        self.assertTrue(scheduler._try_admit_prefill(runtime, 4000))
        self.assertTrue(runtime.admitted_prefill)
        self.assertNotIn(1, scheduler.prefill_admission_dirty)


class DecodeSelectionFilterTest(_PChunkHolderGuard, unittest.TestCase):
    """P1 decode 选择期过滤三态 + 决策确定性 + 零副作用。"""

    def _costs(self):
        return tuple(
            SimpleNamespace(instance_index=i, per_die_delta_ns=30 - 10 * i,
                            weighted_distance=0.0)
            for i in range(3))

    def _runtime(self, tokens):
        return _make_runtime("rx", "sx", prefill_context_tokens=tokens,
                             final_context_tokens=tokens)

    def test_three_states_and_zero_side_effects(self):
        scheduler = _make_scheduler()
        manager = scheduler.kv_manager
        avail = _occupancy_tokens(manager)
        tokens_s = _tokens_for_bytes(int(avail * 0.3))
        _kv_admit_prefill(manager, "sx", 0, tokens_s, 100, "rx")
        runtime = self._runtime(_tokens_for_bytes(int(avail * 0.55)))
        costs = self._costs()

        def frozen_state():
            return (manager.hbm_snapshots(),
                    tuple(manager.events_since(0)),
                    tuple(scheduler.capacity_epoch),
                    tuple(scheduler.online_log_rows),
                    manager.session_snapshot("sx"))

        # 状态 1:原选可行 -> 原样返回,零日志(决策零扰动)。
        before = frozen_state()
        self.assertEqual(
            scheduler._filter_decode_selection(runtime, 1, costs, 100), 1)
        self.assertEqual(frozen_state(), before)

        # 状态 2:原选 inst1 被 ACTIVE 的 sy 占满(不可行);inst0(增量)
        # 与 inst2(全量)可行 -> 取可行集内 per_die_delta_ns 最小 = inst2。
        tokens_y = _tokens_for_bytes(
            avail - _shard_per_rank(_tokens_for_bytes(int(avail * 0.55))) + 1)
        _kv_admit_prefill(manager, "sy", 1, tokens_y, 110, "ry")
        self.assertEqual(
            scheduler._filter_decode_selection(runtime, 1, costs, 110), 2)
        probe_rows = [row for row in scheduler.online_log_rows
                      if row["kind"] == "decode_admission_probe"]
        self.assertEqual(len(probe_rows), 1)
        self.assertEqual(
            probe_rows[0]["decision"]["admission_probe"]["outcome"],
            "reselected_infeasible_first")
        self.assertEqual(
            probe_rows[0]["decision"]["admission_probe"]
            ["selected_instance_index"], 2)

        # 状态 3:全部实例占满 -> 保持原选,交给 P0-1 阻塞-重试路径。
        tokens_z = _tokens_for_bytes(
            avail - _shard_per_rank(_tokens_for_bytes(int(avail * 0.55))) + 1)
        _kv_admit_prefill(manager, "sz", 0, tokens_z, 120, "rz")
        _kv_admit_prefill(manager, "sw", 2, tokens_z, 120, "rw")
        before = frozen_state()
        self.assertEqual(
            scheduler._filter_decode_selection(runtime, 1, costs, 120), 1)
        probe_rows = [row for row in scheduler.online_log_rows
                      if row["kind"] == "decode_admission_probe"]
        self.assertEqual(len(probe_rows), 2)
        self.assertEqual(
            probe_rows[-1]["decision"]["admission_probe"]["outcome"],
            "kept_all_infeasible")

        # 零副作用:状态 3 的全失败探测不改变任何 KV/纪元/事件状态
        # (hbm/session 快照、capacity_epoch、KV 事件流逐项不变)。
        after = (
            manager.hbm_snapshots(),
            tuple(manager.events_since(0)),
            tuple(scheduler.capacity_epoch),
            manager.session_snapshot("sx"),
        )
        self.assertEqual(after[:3], before[:3])
        self.assertEqual(after[3], before[4])


class PrefillSelectionFilterTest(_PChunkHolderGuard, unittest.TestCase):
    """P1 prefill 选择期过滤三态(全上下文保守口径)。"""

    def _snapshots(self):
        return [
            SimpleNamespace(instance_index=0, ordering_key=(2, 0, 0)),
            SimpleNamespace(instance_index=1, ordering_key=(0, 0, 0)),
            SimpleNamespace(instance_index=2, ordering_key=(1, 0, 0)),
        ]

    def test_three_states(self):
        scheduler = _make_scheduler()
        manager = scheduler.kv_manager
        avail = _occupancy_tokens(manager)
        tokens_p = _tokens_for_bytes(int(avail * 0.55))
        runtime = _make_runtime("rx", "sx", prefill_context_tokens=tokens_p)
        snapshots = self._snapshots()

        # 状态 1:原选 inst1 可行 -> 原样返回,零日志。
        self.assertEqual(
            scheduler._filter_prefill_selection(
                runtime, snapshots, 1, 100), 1)
        self.assertEqual(scheduler.online_log_rows, [])

        # 状态 2:inst1 占满 -> 幸存 {inst0, inst2},按 ordering_key 取
        # 最小 = inst2(键 (1,0,0) < (2,0,0))。
        tokens_y = _tokens_for_bytes(avail - _shard_per_rank(tokens_p) + 1)
        _kv_admit_prefill(manager, "sy", 1, tokens_y, 110, "ry")
        self.assertEqual(
            scheduler._filter_prefill_selection(
                runtime, snapshots, 1, 110), 2)
        probe_rows = [row for row in scheduler.online_log_rows
                      if row["kind"] == "prefill_admission_probe"]
        self.assertEqual(len(probe_rows), 1)
        self.assertEqual(
            probe_rows[0]["decision"]["admission_probe"]["outcome"],
            "reselected_infeasible_first")

        # 状态 3:全部占满 -> 幸存集空 -> 保持原选。
        _kv_admit_prefill(manager, "sz", 0, tokens_y, 120, "rz")
        _kv_admit_prefill(manager, "sw", 2, tokens_y, 120, "rw")
        self.assertEqual(
            scheduler._filter_prefill_selection(
                runtime, snapshots, 1, 120), 1)
        probe_rows = [row for row in scheduler.online_log_rows
                      if row["kind"] == "prefill_admission_probe"]
        self.assertEqual(
            probe_rows[-1]["decision"]["admission_probe"]["outcome"],
            "kept_all_infeasible")


class ProbeSemanticsTest(_PChunkHolderGuard, unittest.TestCase):
    """预检口径:source==target 按增量、source!=target 按全量(镜像
    move_prefill_to_decode);RESIDENT 之外的会话状态不给增量信用。"""

    def test_incremental_vs_full_requirement(self):
        scheduler = _make_scheduler()
        manager = scheduler.kv_manager
        avail = _occupancy_tokens(manager)
        tokens_s = _tokens_for_bytes(int(avail * 0.3))
        tokens_f = _tokens_for_bytes(int(avail * 0.9))
        _kv_admit_prefill(manager, "sx", 0, tokens_s, 100, "rx")
        runtime = _make_runtime("rx", "sx", prefill_context_tokens=tokens_f,
                                final_context_tokens=tokens_f)
        snapshot = manager.session_snapshot("sx")
        self.assertEqual(snapshot.state, RESIDENT)
        self.assertEqual(snapshot.instance_index, 0)
        # inst0:增量需求 = final - 已驻留 -> 可行;inst1/inst2:全量
        # > 空余(0.9 > 0.7) -> 不可行。
        final_shards = kv_cache_shard_bytes_for_tokens(MODEL, tokens_f, TP)
        incremental = tuple(
            want - have for want, have in zip(
                final_shards, snapshot.shard_bytes))
        # inst0(会话驻留,余 0.7):增量 0.6 可行、全量 0.9 不可行
        # ——增量/全量口径差异正体现在这里。
        self.assertTrue(scheduler._probe_instance_capacity(0, incremental))
        self.assertFalse(scheduler._probe_instance_capacity(0, final_shards))
        # 空实例 inst2:全量 0.9 可行;超容量需求不可行——口径边界即
        # 过滤边界。
        over = _tokens_for_bytes(avail + 1)
        over_shards = kv_cache_shard_bytes_for_tokens(MODEL, over, TP)
        self.assertTrue(scheduler._probe_instance_capacity(2, final_shards))
        self.assertFalse(scheduler._probe_instance_capacity(2, over_shards))


class DecodeEvictionSerializationTest(_PChunkHolderGuard,
                                      unittest.TestCase):
    """问题 2A 顺带修复(2026-09-05,方案 §4/T-E1):decode 决策行新增
    decode_target_evictions 快照序列化(与 wscllm 同名同构;face 的 decode
    决策在 _try_admit_waiting_decodes 内发射,快照天然含跨 attempt 累积),
    发射后置空核销;prefill 行原硬编码 [] 改真实序列化(face 准入链无
    reserve_request_capacity,发射时点恒空列表,数值不变);completion 行
    内容不受影响。
    """

    @staticmethod
    def _eviction(time_ns, phase, reason, victim_session):
        return EvictionRecord(
            time_ns=time_ns, phase=phase, reason=reason,
            trigger_request_id="rx", victim_session_id=victim_session,
            victim_instance_index=1, victim_last_completion_ns=0,
            context_tokens=7, shard_bytes=(11, 13))

    @classmethod
    def _expected_dict(cls, record):
        return {
            "time_ns": record.time_ns,
            "phase": record.phase,
            "reason": record.reason,
            "trigger_request_id": record.trigger_request_id,
            "victim_session_id": record.victim_session_id,
            "victim_instance_index": record.victim_instance_index,
            "victim_last_completion_ns":
                record.victim_last_completion_ns,
            "context_tokens": record.context_tokens,
            "shard_bytes": list(record.shard_bytes),
        }

    @staticmethod
    def _decode_ready_runtime(scheduler, tokens):
        """rx 会话预置于 inst0,decode target = 空 inst1(准入链自然逐出
        为空,新字段内容完全由注入记录决定)。"""
        _kv_admit_prefill(
            scheduler.kv_manager, "sx", 0, tokens, 100, "rx")
        runtime = _make_runtime(
            "rx", "sx", prefill_context_tokens=tokens,
            final_context_tokens=tokens)
        runtime.prefill_instance_index = 0
        runtime.decode_instance_index = 1
        runtime.waiting_decode_admission = True
        scheduler.waiting_decode_admissions[1].append(runtime)
        return runtime

    def test_decode_row_serializes_injected_then_clears(self):
        scheduler = _make_scheduler()
        tokens_x = _tokens_for_bytes(
            int(_occupancy_tokens(scheduler.kv_manager) * 0.6))
        runtime = self._decode_ready_runtime(scheduler, tokens_x)
        e4 = self._eviction(400, "prefill_decode",
                            "prefill_decode_capacity", "victim_a")
        e5 = self._eviction(500, "decode",
                            "decode_growth_capacity", "victim_b")
        runtime.decode_target_evictions = (e4, e5)

        scheduler._try_admit_waiting_decodes(1000)
        self.assertFalse(runtime.waiting_decode_admission)

        decode_rows = [row for row in scheduler.online_log_rows
                       if row["kind"] == "decode"]
        self.assertEqual(len(decode_rows), 1)
        serialized = decode_rows[0]["decision"]["decode_target_evictions"]
        # 注入的 E4/E5 全字段(9 属性)序列化,与 _eviction_dict 逐项相等。
        self.assertEqual(
            serialized,
            [self._expected_dict(e4), self._expected_dict(e5)])
        # 发射后置空核销(镜像 completion 对 completion_evictions 的
        # 先序列化后置空语义;face_online_scheduler 的 _on_decode_complete
        # 完成即置空保留为幂等兜底,2026-10-01 去行号锚防漂移)。
        self.assertEqual(runtime.decode_target_evictions, ())

    def test_completion_row_unaffected_by_decode_row_serialization(self):
        scheduler = _make_scheduler()
        tokens_x = _tokens_for_bytes(
            int(_occupancy_tokens(scheduler.kv_manager) * 0.6))
        runtime = self._decode_ready_runtime(scheduler, tokens_x)
        e4 = self._eviction(400, "prefill_decode",
                            "prefill_decode_capacity", "victim_a")
        runtime.decode_target_evictions = (e4,)

        scheduler._try_admit_waiting_decodes(1000)
        # decode 行发射后置空:completion 边界的 M4 置空成为幂等兜底,
        # completion 行内容(completion_evictions 等三键)不受影响。
        scheduler.runtime_by_request_id = {"rx": runtime}
        scheduler.completed_requests = 0
        scheduler._on_decode_complete("rx", 2000)

        rows = scheduler.online_log_rows
        completion_rows = [row for row in rows
                           if row["kind"] == "completion"]
        self.assertEqual(len(completion_rows), 1)
        decision = completion_rows[0]["decision"]
        self.assertEqual(
            set(decision),
            {"kv_state_after_completion", "kv_instance_after_completion",
             "completion_evictions"})
        self.assertNotIn("decode_target_evictions", decision)
        # decode 行不被 completion 补写/改写(每请求恰一条)。
        decode_rows = [row for row in rows if row["kind"] == "decode"]
        self.assertEqual(len(decode_rows), 1)
        self.assertEqual(
            decode_rows[0]["decision"]["decode_target_evictions"],
            [self._expected_dict(e4)])

    def test_prefill_row_serializes_holder_not_hardcoded(self):
        scheduler = _make_scheduler()
        # _emit_admission 的最小装配(graph/发射账本打桩,决策行为真)。
        scheduler._batch = {"assignments": [], "delivery_sequence": 0}
        scheduler._emitted_by_delivery = {0: {"requests": []}}
        scheduler.ledger_issued = {}
        scheduler.graph = SimpleNamespace(
            emit_admission_batch=lambda plan: None)

        def _emit(runtime):
            scheduler._emit_admission(runtime, 1500)
            return scheduler.online_log_rows[-1]

        runtime = _make_runtime("rx", "sx", prefill_context_tokens=64)
        runtime.prefill_instance_index = 0
        runtime.prefill_assignment_key = (0, 0, 0)
        runtime.history_recompute_tokens = 0  # _emit_admission 读取该聚合字段
        # 初始()(真实准入发射时点的取值)→ 数值不变:仍为空列表。
        self.assertEqual(_emit(runtime)["decision"]
                         ["decode_target_evictions"], [])
        # holder 非空(假想未来准入链预占)→ 真实序列化,证明不再是
        # 硬编码 [](语义正确化)。
        e4 = self._eviction(400, "prefill_decode",
                            "prefill_decode_capacity", "victim_a")
        runtime.decode_target_evictions = (e4,)
        self.assertEqual(_emit(runtime)["decision"]
                         ["decode_target_evictions"],
                         [self._expected_dict(e4)])


class CapacitySignalTouchpointTest(_PChunkHolderGuard, unittest.TestCase):
    """信号触点回归锁(2026-10-20 方案一风险一)。

    审计表(逐容量释放路径 -> 调度器调用点 -> 信号)结论"无缺口"的
    回归锁:逐出完成 / retirement / workspace 外部视图回落 / drain 准入
    四条预期族路径,在最小内核上锁定 capacity_epoch 推进或 dirty 重试门
    ——信号缺失即 2026-08-31 死端家族,任何一条回退本组即红。"""

    def test_eviction_completion_path_advances_epoch(self):
        # 逐出完成路径:blocker 完成 -> 冷会话可逐 -> 准入成功路径的
        # _note_capacity_change 推进目标实例容量纪元。
        scheduler = _make_scheduler()
        manager = scheduler.kv_manager
        avail = _occupancy_tokens(manager)
        tokens_x = _tokens_for_bytes(int(avail * 0.6))
        tokens_y = _tokens_for_bytes(avail - _shard_per_rank(tokens_x) + 1)
        _kv_admit_prefill(manager, "sy", 1, tokens_y, 100, "ry")
        runtime = _make_runtime("rx", "sx", prefill_context_tokens=tokens_x)
        runtime.prefill_instance_index = 1
        self.assertFalse(scheduler._try_admit_prefill(runtime, 3000))
        self.assertEqual(scheduler.capacity_epoch[1], 0)
        manager.mark_complete("sy", 3500, "ry")
        self.assertTrue(scheduler._try_admit_prefill(runtime, 4000))
        self.assertGreater(scheduler.capacity_epoch[1], 0)

    def test_retirement_path_advances_epoch(self):
        # retirement 路径:terminal turn 会话整体删除 ->
        # _note_capacity_change(released_instance_index) 恰推进 1。
        scheduler = _make_scheduler()
        manager = scheduler.kv_manager
        _kv_admit_prefill(manager, "sx", 2, 64, 100, "rx")
        manager.mark_complete("sx", 200, "rx")
        runtime = _make_runtime("rx", "sx", prefill_context_tokens=64)
        runtime.decode_instance_index = 2
        scheduler.runtime_by_request_id = {"rx": runtime}
        scheduler._runtime_index = {"rx": 0}
        scheduler.runtimes = [runtime]
        scheduler.next_request = [None]  # terminal turn
        scheduler.completed_requests = 1
        scheduler.graph = SimpleNamespace(
            retire_completion_gate=lambda session_id: None)
        scheduler._batch = {"future_alarms": []}
        epoch_before = scheduler.capacity_epoch[2]
        scheduler._on_request_complete("rx", 300)
        self.assertEqual(scheduler.capacity_epoch[2], epoch_before + 1)
        self.assertIsNone(manager.session_snapshot("sx"))

    def test_workspace_view_fallback_advances_epoch_and_tracks_peak(self):
        # workspace 外部视图回落路径:set_external_workspace_view 有变更
        # (含 bytes 回落释放容量)-> _note_capacity_change;同 revision 同
        # bytes 无变更 -> 纪元不动;风险三峰值采集维护运行最大值。
        scheduler = _make_scheduler()
        scheduler.workspace_admission = "required"
        scheduler._workspace_rank_to_instance = {
            rank: instance.index
            for instance in scheduler.topology.instances
            for rank in instance.ranks}
        ranks = sorted(scheduler.kv_manager.rank_ids)

        def _delta(revision, bytes_by_rank):
            return {"workspace_state": {
                "protocol_version": 1,
                "ranks": [
                    {"rank": rank, "workspace_revision": revision,
                     "active_workspace_bytes": bytes_by_rank[rank]}
                    for rank in ranks]}}

        scheduler._apply_delivery_extension(
            _delta(1, {rank: 1000 for rank in ranks}))
        self.assertEqual(list(scheduler.capacity_epoch), [1] * 3)
        self.assertEqual(scheduler._workspace_peak_bytes_by_rank,
                         {rank: 1000 for rank in ranks})
        # 无变更交付:纪元与峰值都不动。
        scheduler._apply_delivery_extension(
            _delta(1, {rank: 1000 for rank in ranks}))
        self.assertEqual(list(scheduler.capacity_epoch), [1] * 3)
        # revision 前进 + bytes 回落(容量释放形态):纪元再推进,
        # 峰值保持历史最大(回落不抹峰值——对账口径是峰值不是末值)。
        scheduler._apply_delivery_extension(
            _delta(2, {rank: 600 for rank in ranks}))
        self.assertEqual(list(scheduler.capacity_epoch), [2] * 3)
        self.assertEqual(scheduler._workspace_peak_bytes_by_rank,
                         {rank: 1000 for rank in ranks})

    def test_drain_admission_path_marks_dirty_and_logs_blocked(self):
        # drain 准入路径:prefill drain -> waiting decode 登记 + dirty +
        # 同 tick try_admit_waiting_decodes;全实例不可行时阻塞重排队,
        # 阻塞行即增富观测的风险一数据源。
        scheduler = _make_scheduler()
        manager = scheduler.kv_manager
        scheduler.instance_graph = WeightedInstanceGraph(scheduler.topology)
        scheduler._decode_tie_counter = DecodeTieCounter()
        scheduler.cost_oracle = None
        avail = _occupancy_tokens(manager)
        tokens_p = _tokens_for_bytes(int(avail * 0.45))
        tokens_f = _tokens_for_bytes(int(avail * 0.9))
        tokens_fill = _tokens_for_bytes(int(avail * 0.5))
        # rx 会话预置于 inst0(0.45);sy 钉满 inst0 余量;sz/sw 钉住
        # inst1/inst2(0.5 各一)——增量(0.45)与全量(0.9)无处可放,
        # 任何 decode 实例选择都阻塞(ACTIVE 不可逐出)。
        _kv_admit_prefill(manager, "sx", 0, tokens_p, 100, "rx")
        _kv_admit_prefill(manager, "sy", 0, tokens_fill, 110, "ry")
        _kv_admit_prefill(manager, "sz", 1, tokens_fill, 110, "rz")
        _kv_admit_prefill(manager, "sw", 2, tokens_fill, 110, "rw")
        runtime = _make_runtime(
            "rx", "sx", prefill_context_tokens=tokens_p,
            final_context_tokens=tokens_f)
        runtime.prefill_instance_index = 0
        scheduler.instances[0].qp.append(runtime)
        scheduler.runtime_by_request_id = {"rx": runtime}

        scheduler._on_prefill_drain("rx", 1000)
        selected = runtime.decode_instance_index
        self.assertTrue(runtime.waiting_decode_admission)
        self.assertIn(runtime, scheduler.waiting_decode_admissions[selected])
        # drain 边界信号:选定实例即置 dirty(阻塞重排队后保持置位)。
        self.assertIn(selected, scheduler.decode_admission_dirty)
        blocked_rows = [row for row in scheduler.online_log_rows
                        if row["kind"] == "admission_blocked"]
        self.assertEqual(len(blocked_rows), 1)
        self.assertEqual(blocked_rows[0]["request_id"], "rx")
        self.assertEqual(blocked_rows[0]["decision"]["stage"], "decode")
        self.assertEqual(
            blocked_rows[0]["decision"]["instance_index"], selected)


class AdmissionBlockedEnrichmentTest(_PChunkHolderGuard, unittest.TestCase):
    """阻塞行增富(2026-10-20 方案一风险一):admission_blocked 决策行
    携带 stage/instance_index/capacity_epoch/kv_free_bytes_by_rank/
    required_bytes_by_rank,数值与只读账本逐项相等——无会话全量口径与
    LOCAL_HBM(RESIDENT)同实例增量口径两形态。"""

    def test_prefill_blocked_row_full_requirement_without_session(self):
        # 无会话形态:rx 的会话从未准入(session_snapshot 为 None),
        # 需求 = prefill_context 全量分片;空闲 = 逐 rank fit 余量。
        scheduler = _make_scheduler()
        manager = scheduler.kv_manager
        avail = _occupancy_tokens(manager)
        tokens_x = _tokens_for_bytes(int(avail * 0.6))
        tokens_y = _tokens_for_bytes(avail - _shard_per_rank(tokens_x) + 1)
        _kv_admit_prefill(manager, "sy", 1, tokens_y, 100, "ry")
        runtime = _make_runtime("rx", "sx", prefill_context_tokens=tokens_x)
        runtime.prefill_instance_index = 1

        self.assertFalse(scheduler._try_admit_prefill(runtime, 3000))
        blocked_rows = [row for row in scheduler.online_log_rows
                        if row["kind"] == "admission_blocked"]
        self.assertEqual(len(blocked_rows), 1)
        row = blocked_rows[0]
        self.assertEqual(row["request_id"], "rx")
        decision = row["decision"]
        self.assertEqual(decision["stage"], "prefill")
        self.assertEqual(decision["instance_index"], 1)
        # 无逐出阻塞不推进纪元:观测行携带当前值 0。
        self.assertEqual(decision["capacity_epoch"], 0)
        self.assertEqual(
            decision["kv_free_bytes_by_rank"],
            {rank: manager.workspace_fit_remaining_bytes(rank)
             for rank in scheduler.topology.instances[1].ranks})
        full = kv_cache_shard_bytes_for_tokens(MODEL, tokens_x, TP)
        self.assertEqual(
            decision["required_bytes_by_rank"],
            {rank: shard for rank, shard in zip(
                scheduler.topology.instances[1].ranks, full)})

    def test_decode_blocked_row_incremental_requirement_local_session(self):
        # LOCAL_HBM 同实例增量形态:rx 会话 RESIDENT 于 decode target
        # inst0,需求 = final - 已驻留 shard_bytes(逐 rank)。
        scheduler = _make_scheduler()
        manager = scheduler.kv_manager
        avail = _occupancy_tokens(manager)
        tokens_p = _tokens_for_bytes(int(avail * 0.45))
        tokens_f = _tokens_for_bytes(int(avail * 0.9))
        tokens_fill = _tokens_for_bytes(int(avail * 0.5))
        _kv_admit_prefill(manager, "sx", 0, tokens_p, 100, "rx")
        _kv_admit_prefill(manager, "sy", 0, tokens_fill, 110, "ry")
        runtime = _make_runtime(
            "rx", "sx", prefill_context_tokens=tokens_p,
            final_context_tokens=tokens_f)
        runtime.prefill_instance_index = 0
        runtime.decode_instance_index = 0
        runtime.waiting_decode_admission = True
        scheduler.waiting_decode_admissions[0].append(runtime)

        scheduler._try_admit_waiting_decodes(1000)
        self.assertTrue(runtime.waiting_decode_admission)
        blocked_rows = [row for row in scheduler.online_log_rows
                        if row["kind"] == "admission_blocked"]
        self.assertEqual(len(blocked_rows), 1)
        decision = blocked_rows[0]["decision"]
        self.assertEqual(decision["stage"], "decode")
        self.assertEqual(decision["instance_index"], 0)
        self.assertEqual(
            decision["kv_free_bytes_by_rank"],
            {rank: manager.workspace_fit_remaining_bytes(rank)
             for rank in scheduler.topology.instances[0].ranks})
        resident = manager.session_snapshot("sx").shard_bytes
        final = kv_cache_shard_bytes_for_tokens(MODEL, tokens_f, TP)
        self.assertEqual(
            decision["required_bytes_by_rank"],
            {rank: want - have for rank, want, have in zip(
                scheduler.topology.instances[0].ranks, final, resident)})
        # 增量口径数值健全:每 rank required > 0 且确实放不下
        # (阻塞为真,非观测假象)。
        self.assertTrue(all(
            value > 0 for value in decision["required_bytes_by_rank"].values()))
        self.assertTrue(all(
            free < required for free, required in zip(
                decision["kv_free_bytes_by_rank"].values(),
                decision["required_bytes_by_rank"].values())))


class StreakAndRunEndGuardTest(_PChunkHolderGuard, unittest.TestCase):
    """streak 簿记与 verify_run_end 终检(2026-10-20 方案一风险一):
    连续阻塞计数/历史最大/进展清账;run 结束仍阻塞即 raise;直方图由
    历史最大值表产出。"""

    @staticmethod
    def _complete_run_state(scheduler):
        """把 __new__ 夹具补齐到可通过 verify_run_end 的最小完整run态
        (基类协议计数 + 子类收尾断言所需的空/已核销容器)。"""
        scheduler.ack_count = 1
        scheduler.delivery_count = 1
        scheduler.last_applied_sequence = 0
        scheduler._delivery_reply_cache = None
        scheduler.in_flight = {}
        scheduler.expected_request_count = 1
        scheduler.arrived_request_count = 1
        scheduler.completed_request_count = 1
        scheduler.unseen_request_ids = set()
        scheduler._emitted_by_delivery = {}
        scheduler._acked_through = 0
        scheduler._acked_out_of_order = set()
        scheduler._provisional_kv_actions = set()
        scheduler.sensing_enabled = False
        scheduler.completed_requests = 1
        scheduler.runtime_by_request_id = {}
        scheduler._runtime_index = {}
        scheduler.runtimes = [None]
        scheduler._pending_first_steps = {}
        scheduler.arrival_heap = []
        scheduler.graph = SimpleNamespace(completion_gates=set())

    def test_streak_lifecycle_and_histogram(self):
        scheduler = _make_scheduler()
        manager = scheduler.kv_manager
        avail = _occupancy_tokens(manager)
        tokens_x = _tokens_for_bytes(int(avail * 0.6))
        tokens_y = _tokens_for_bytes(avail - _shard_per_rank(tokens_x) + 1)
        _kv_admit_prefill(manager, "sy", 1, tokens_y, 100, "ry")
        runtime = _make_runtime("rx", "sx", prefill_context_tokens=tokens_x)
        runtime.prefill_instance_index = 1

        # 两个决策边界连续阻塞:streak=2,max=2,open 挂起。
        self.assertFalse(scheduler._try_admit_prefill(runtime, 3000))
        self.assertFalse(scheduler._try_admit_prefill(runtime, 3001))
        self.assertEqual(scheduler._admission_blocked_streak, {"rx": 2})
        self.assertEqual(scheduler._admission_blocked_streak_max, {"rx": 2})
        self.assertEqual(scheduler._admission_blocked_open,
                         {"rx": ("prefill", 3001)})

        # blocker 完成 -> 准入成功即进展清账(当前表清,历史最大保留)。
        manager.mark_complete("sy", 3500, "ry")
        self.assertTrue(scheduler._try_admit_prefill(runtime, 4000))
        self.assertEqual(scheduler._admission_blocked_streak, {})
        self.assertEqual(scheduler._admission_blocked_open, {})
        self.assertEqual(scheduler._admission_blocked_streak_max, {"rx": 2})

        # 完整 run 收尾:直方图 {"2": 1};off 模式无对账(保持 None)。
        manager.mark_complete("sx", 5000, "rx")
        self._complete_run_state(scheduler)
        scheduler.verify_run_end()
        self.assertEqual(scheduler.admission_blocked_streak_histogram,
                         {"2": 1})
        self.assertIsNone(scheduler.workspace_floor_reconciliation)

    def test_verify_run_end_raises_on_open_block(self):
        scheduler = _make_scheduler()
        self._complete_run_state(scheduler)
        scheduler._admission_blocked_open = {"rq7": ("decode", 42)}
        with self.assertRaises(RuntimeError) as caught:
            scheduler.verify_run_end()
        message = str(caught.exception)
        self.assertIn("admission-blocked without progress", message)
        self.assertIn("rq7", message)
        self.assertIn("decode", message)

    def test_verify_run_end_empty_histogram_without_blocks(self):
        scheduler = _make_scheduler()
        self._complete_run_state(scheduler)
        scheduler.verify_run_end()
        self.assertEqual(scheduler.admission_blocked_streak_histogram, {})
        self.assertIsNone(scheduler.workspace_floor_reconciliation)


class WorkspaceFloorReconciliationTest(_PChunkHolderGuard,
                                        unittest.TestCase):
    """风险三最小对账(2026-10-20 方案一):workspace 峰值 vs prospective
    证书地板——越界即 raise;峰值内产出逐 rank slack 字典;非 required
    模式保持 None。"""

    def _required_scheduler(self, floor_bytes):
        scheduler = _make_scheduler()
        scheduler.workspace_admission = "required"
        scheduler._workspace_rank_to_instance = {
            rank: instance.index
            for instance in scheduler.topology.instances
            for rank in instance.ranks}
        ranks = sorted(scheduler.kv_manager.rank_ids)
        scheduler.kv_manager = SessionKVCacheManager(
            scheduler.topology, MODEL,
            prospective_workspace_bytes_by_rank={
                rank: floor_bytes for rank in ranks})
        return scheduler, ranks

    @staticmethod
    def _apply_view(scheduler, ranks, revision, bytes_by_rank):
        scheduler._apply_delivery_extension({"workspace_state": {
            "protocol_version": 1,
            "ranks": [
                {"rank": rank, "workspace_revision": revision,
                 "active_workspace_bytes": bytes_by_rank[rank]}
                for rank in ranks]}})

    def _complete_run_state(self, scheduler):
        StreakAndRunEndGuardTest._complete_run_state(scheduler)

    def test_peak_exceeds_floor_raises(self):
        scheduler, ranks = self._required_scheduler(floor_bytes=3000)
        self._apply_view(scheduler, ranks, 1, {rank: 4000 for rank in ranks})
        self._complete_run_state(scheduler)
        with self.assertRaises(RuntimeError) as caught:
            scheduler.verify_run_end()
        self.assertIn(
            "workspace peak exceeded prospective certificate floor",
            str(caught.exception))

    def test_peak_within_floor_produces_slack(self):
        scheduler, ranks = self._required_scheduler(floor_bytes=5000)
        # 只有 rank0 出现过 workspace 视图(其余 rank 峰值缺省 0——
        # 对账字典仍覆盖证书全部 rank)。
        bytes_by_rank = {rank: (4000 if rank == ranks[0] else 0)
                         for rank in ranks}
        self._apply_view(scheduler, ranks, 1, bytes_by_rank)
        self._complete_run_state(scheduler)
        scheduler.verify_run_end()
        expected = {
            "peak_bytes_by_rank": {
                str(rank): (4000 if rank == ranks[0] else 0)
                for rank in ranks},
            "floor_bytes_by_rank": {str(rank): 5000 for rank in ranks},
            "slack_bytes_by_rank": {
                str(rank): (1000 if rank == ranks[0] else 5000)
                for rank in ranks},
        }
        self.assertEqual(scheduler.workspace_floor_reconciliation, expected)

    def test_off_mode_stays_none(self):
        scheduler = _make_scheduler()
        scheduler.workspace_admission = "off"
        self._complete_run_state(scheduler)
        scheduler._workspace_peak_bytes_by_rank = {0: 4000, 1: 4000}
        scheduler.verify_run_end()
        self.assertIsNone(scheduler.workspace_floor_reconciliation)


class AdmissionLivenessAuditToolTest(_PChunkHolderGuard, unittest.TestCase):
    """离线活性审计器四态单测(online/verify/admission_liveness_audit.py,
    2026-10-20 方案一):PASS(无阻塞行)/PASS(短 streak 或不可行
    streak)/FAIL(>=3 连续可行 streak)/SKIP(增富字段缺失的长
    streak);外加 run 目录自动定位(bridge/ 下探)。直接驱动 main()。"""

    @staticmethod
    def _run_main(argv):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = admission_liveness_audit.main(argv)
        return code, out.getvalue()

    @staticmethod
    def _write_log(directory, rows):
        path = directory / "online_decision_log.jsonl"
        with open(path, "w", encoding="utf-8") as sink:
            for row in rows:
                sink.write(json.dumps(row, sort_keys=True) + "\n")
        return path

    @staticmethod
    def _progress(seq, request_id="r1"):
        return {"seq": seq, "tick": seq, "kind": "decode",
                "request_id": request_id, "priority": 0, "decision": {}}

    @staticmethod
    def _blocked(seq, *, request_id="r1", free=1000, required=500,
                 enrich=True):
        decision = ({"stage": "decode", "instance_index": 0,
                     "capacity_epoch": 0,
                     "kv_free_bytes_by_rank": {"0": free, "1": free},
                     "required_bytes_by_rank": {"0": required,
                                                "1": required}}
                    if enrich else {})
        return {"seq": seq, "tick": seq, "kind": "admission_blocked",
                "request_id": request_id, "priority": 0,
                "decision": decision}

    def test_pass_no_blocked_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_log(
                Path(tmp), [self._progress(1), self._progress(2)])
            code, out = self._run_main([str(path)])
            self.assertEqual(code, 0)
            self.assertIn("PASS", out)
            self.assertIn("no admission_blocked rows", out)

    def test_pass_short_streak_below_threshold(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_log(Path(tmp), [
                self._blocked(1), self._blocked(2), self._progress(3)])
            code, out = self._run_main([str(path)])
            self.assertEqual(code, 0)
            self.assertIn("PASS", out)
            self.assertIn("no free-capacity deadlock streak", out)

    def test_pass_infeasible_streak_not_a_violation(self):
        # 长 streak 但空闲不足(free < required)——真缺容量不是死端。
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_log(Path(tmp), [
                self._blocked(1, free=100, required=500),
                self._blocked(2, free=100, required=500),
                self._blocked(3, free=100, required=500),
                self._blocked(4, free=100, required=500)])
            code, out = self._run_main([str(path)])
            self.assertEqual(code, 0)
            self.assertIn("PASS", out)

    def test_fail_feasible_streak_at_threshold(self):
        # 同请求 3 个连续可行阻塞行(free >= required):死端可判定形态。
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_log(Path(tmp), [
                self._blocked(1), self._blocked(2), self._blocked(3)])
            code, out = self._run_main([str(path)])
            self.assertEqual(code, 1)
            self.assertIn("FAIL", out)
            self.assertIn("r1", out)

    def test_fail_is_cut_by_progress_row(self):
        # 进展行(decode)切断 streak:2+2 不达阈值 -> PASS(进度语义)。
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_log(Path(tmp), [
                self._blocked(1), self._blocked(2), self._progress(3),
                self._blocked(4), self._blocked(5)])
            code, _ = self._run_main([str(path)])
            self.assertEqual(code, 0)

    def test_skip_missing_enrichment_fields(self):
        # 增富字段缺失的旧构建日志:长 streak 明确 SKIP,绝不误判。
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_log(Path(tmp), [
                self._blocked(1, enrich=False),
                self._blocked(2, enrich=False),
                self._blocked(3, enrich=False)])
            code, out = self._run_main([str(path)])
            self.assertEqual(code, 0)
            self.assertIn("SKIP", out)

    def test_run_dir_autolocates_bridge_log(self):
        # run 目录自动定位:日志在 bridge/ 子目录下也能找到(runner 接线
        # 依赖该形态;归档前审计时日志尚未移入 results/)。
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "bridge").mkdir()
            self._write_log(run_dir / "bridge", [self._progress(1)])
            code, out = self._run_main([str(run_dir)])
            self.assertEqual(code, 0)
            self.assertIn("PASS", out)


if __name__ == "__main__":
    unittest.main()
