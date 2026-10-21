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

测试组织说明(与 online/ 既有测试同风格):被测方法全部为真实实现;
协作者(kv_manager/instances/runtimes)为真实对象;仅容器经
FaceOnlineScheduler.__new__ 手工装配聚焦属性(完整构造的集成路径由
runner 冒烟覆盖,见执行文档批次3)。KV 状态一律经与真实准入路径相同
的 prepare_history/grow_prefill/mark_complete 调用建立。

运行:cd sh_test_mesh/workload/llama2_7b_inference &&
      python3 online/test_admission_liveness.py   （或 pytest 同路径）
"""
import json
import os
import sys
import unittest
from collections import deque
from types import SimpleNamespace

_ONLINE_DIR = os.path.dirname(os.path.abspath(__file__))
_WORKLOAD_DIR = os.path.dirname(_ONLINE_DIR)
for _p in (_ONLINE_DIR, _WORKLOAD_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from face_scheduler import (  # noqa: E402
    FaceHardware,
    FaceInstanceSpec,
    build_instances,
)
import online.face_online_scheduler as face_online_scheduler  # noqa: E402
from online.face_online_scheduler import (  # noqa: E402
    FaceOnlineScheduler,
    _OnlineInstanceState,
    _OnlineRequestRuntime,
)
from session_kv_manager import (  # noqa: E402
    LOCAL_HBM,
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
    # B3(2026-09-06):三态 KV 发射面打桩(决策行为真,构图发射零动作;
    # 完整构图链由 test_graph_batch_builder 与 runner 冒烟覆盖)。
    # B4 决策→发射窗口补丁(2026-10-01):准入门新查 has_pending_store_
    # decision,打桩恒 False(无"已决策未发射"窗口)。
    scheduler.graph = SimpleNamespace(
        sync_pending_history_after_evictions=lambda transfers: None,
        emit_eviction_actions=lambda plan, transfers: None,
        has_pending_store_decision=lambda session_id: False,
        note_request_complete=(
            lambda session_id, following, location: None),
        retire_request_state=lambda request_id: None,
        retire_completion_gate=lambda session_id: None,
    )
    scheduler.online_log_count = 0
    scheduler.decision_log_sink = None
    scheduler.online_log_rows = []
    scheduler.ledger_admitted = {}
    scheduler.p_chunk = P_CHUNK
    # 方案一(2026-10-20)风险一/风险三观测簿记(镜像生产构造器;观测
    # 方法另有 getattr 惰性建账兜底,这里显式装配保持夹具与生产同构)。
    scheduler._admission_blocked_streak = {}
    scheduler._admission_blocked_streak_max = {}
    scheduler._admission_blocked_open = {}
    scheduler._workspace_peak_bytes_by_rank = {}
    scheduler.admission_blocked_streak_histogram = None
    scheduler.workspace_floor_reconciliation = None
    face_online_scheduler._P_CHUNK_HOLDER[0] = P_CHUNK
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


class _PChunkHolderReset:
    """测试卫生(L22,2026-09-29):_make_scheduler 按生产 __init__ 同款
    改写模块级全局 face_online_scheduler._P_CHUNK_HOLDER(生产默认
    [0],真实调度器构造时经 :529 按 self.p_chunk 重置)。该全局若不在
    测试结束后复位,同进程 pytest/unittest 顺序收集时泄漏的 128 会改
    变后续测试文件里 _OnlineRequestRuntime.remaining_chunks 的离线默
    认口径——setUp 保存、addCleanup 恢复,消除文件间顺序依赖。"""

    def setUp(self):
        super().setUp()
        holder = face_online_scheduler._P_CHUNK_HOLDER
        self.addCleanup(holder.__setitem__, 0, holder[0])


def _kv_admit_prefill(manager, session_id, instance_index, tokens,
                      now_ns, request_id):
    """经与真实准入相同的调用序列把会话置为 RESIDENT+active。"""
    decision = manager.prepare_history(
        session_id, instance_index, 0, now_ns, request_id,
        required_context_tokens=tokens)
    assert not decision.admission_blocked, "fixture 预放置必须可行"
    growth = manager.grow_prefill(session_id, tokens, now_ns, request_id)
    assert growth.admitted, "fixture 预放置 grow_prefill 必须成功"


def _occupancy_bytes(manager):
    """空实例每 rank 可用 KV 字节(扣模型权重)。"""
    snapshot = manager.hbm_snapshots(0)
    return min(state.remaining_bytes for state in snapshot)


def _pressure_events(manager):
    return [
        event for event in manager.events
        if event.event_type in ("admission_blocked", "admission_retry")
    ]


class DecodeAdmissionLivenessTest(_PChunkHolderReset, unittest.TestCase):
    """用例 A:decode 阻塞即重开重试门;blocker 完成的下一边界准入。"""

    def test_blocked_then_admitted_after_blocker_completes(self):
        scheduler = _make_scheduler()
        manager = scheduler.kv_manager
        avail = _occupancy_bytes(manager)
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


class PrefillAdmissionLivenessTest(_PChunkHolderReset, unittest.TestCase):
    """用例 C:prefill 阻塞即置 dirty;不变状态重试;释放后准入。"""

    def test_blocked_marks_dirty_and_retries_then_admits(self):
        scheduler = _make_scheduler()
        manager = scheduler.kv_manager
        avail = _occupancy_bytes(manager)
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


class DecodeSelectionFilterTest(_PChunkHolderReset, unittest.TestCase):
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
        avail = _occupancy_bytes(manager)
        tokens_s = _tokens_for_bytes(int(avail * 0.3))
        _kv_admit_prefill(manager, "sx", 0, tokens_s, 100, "rx")
        runtime = self._runtime(_tokens_for_bytes(int(avail * 0.55)))
        costs = self._costs()

        def frozen_state():
            return (manager.hbm_snapshots(),
                    tuple(manager.events),
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
            tuple(manager.events),
            tuple(scheduler.capacity_epoch),
            manager.session_snapshot("sx"),
        )
        self.assertEqual(after[:3], before[:3])
        self.assertEqual(after[3], before[4])


class PrefillSelectionFilterTest(_PChunkHolderReset, unittest.TestCase):
    """P1 prefill 选择期过滤三态(M12 后增量口径:会话本地驻留且已在候选
    实例上时按增量需求 = prefill_context 分片 − 已驻留 local_shard_bytes,
    其余无会话/远端/异实例按全量——镜像 prepare_history 的真实准入门)。"""

    def _snapshots(self):
        return [
            SimpleNamespace(instance_index=0, ordering_key=(2, 0, 0)),
            SimpleNamespace(instance_index=1, ordering_key=(0, 0, 0)),
            SimpleNamespace(instance_index=2, ordering_key=(1, 0, 0)),
        ]

    def test_three_states(self):
        scheduler = _make_scheduler()
        manager = scheduler.kv_manager
        avail = _occupancy_bytes(manager)
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


class ProbeSemanticsTest(_PChunkHolderReset, unittest.TestCase):
    """预检口径:source==target 按增量、source!=target 按全量(镜像
    move_prefill_to_decode);非本地驻留(三态 location 之外)的会话状态
    不给增量信用。B2(2026-09-06):RESIDENT 常量换三态 location,增量
    信用 = 驻留前缀层段(local_shard_bytes)。"""

    def test_incremental_vs_full_requirement(self):
        scheduler = _make_scheduler()
        manager = scheduler.kv_manager
        avail = _occupancy_bytes(manager)
        tokens_s = _tokens_for_bytes(int(avail * 0.3))
        tokens_f = _tokens_for_bytes(int(avail * 0.9))
        _kv_admit_prefill(manager, "sx", 0, tokens_s, 100, "rx")
        runtime = _make_runtime("rx", "sx", prefill_context_tokens=tokens_f,
                                final_context_tokens=tokens_f)
        snapshot = manager.session_snapshot("sx")
        # B2:活跃会话恒为全层本地驻留(local_hbm,前缀 = 全部 L 层)。
        self.assertEqual(snapshot.location, LOCAL_HBM)
        self.assertEqual(snapshot.instance_index, 0)
        self.assertEqual(
            snapshot.resident_prefix_layers, MODEL.layers)
        # inst0:增量需求 = final - 已驻留 -> 可行;inst1/inst2:全量
        # > 空余(0.9 > 0.7) -> 不可行。
        final_shards = kv_cache_shard_bytes_for_tokens(MODEL, tokens_f, TP)
        incremental = tuple(
            want - have for want, have in zip(
                final_shards, snapshot.local_shard_bytes))
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


class DecodeEvictionSerializationTest(_PChunkHolderReset, unittest.TestCase):
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
            int(_occupancy_bytes(scheduler.kv_manager) * 0.6))
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
        # 先序列化后置空语义;:1171 M4 置空保留为幂等兜底)。
        self.assertEqual(runtime.decode_target_evictions, ())

    def test_completion_row_unaffected_by_decode_row_serialization(self):
        scheduler = _make_scheduler()
        tokens_x = _tokens_for_bytes(
            int(_occupancy_bytes(scheduler.kv_manager) * 0.6))
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
             "completion_evictions",
             # B3(契约 §3):完成段契约行 + hop 摘要(face 完成路径零
             # 逐出,行恒空;字段在场性要求)。
             "completion_eviction_transfers", "transfer_hop_bytes"})
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


class TieredRestoreAdmissionTest(_PChunkHolderReset, unittest.TestCase):
    """Session 级 Tiered-LRU(2026-09-25 重写):唯一恢复路径经真实
    _try_admit_prefill 接线——REMOTE -> REMOTE_RESTORE(单笔全量回迁
    [0, L));铺垫改用重写后 _evict_session(reason='probe') 整体外迁;
    remaining_chunks 口径不再含 recompute 段(ceil(R/p_chunk) 项删除)。
    KV 状态经真实准入/完成/逐出调用建立。"""

    def _admit_turn(self, scheduler, request_id, session_id, turn,
                    history_tokens, prefill_length, target_instance, now_ns):
        runtime = _make_runtime(
            request_id, session_id, turn=turn,
            prefill_context_tokens=history_tokens + prefill_length,
            final_context_tokens=history_tokens + prefill_length,
            prefill_length=prefill_length,
            history_tokens_before=history_tokens)
        runtime.prefill_instance_index = target_instance
        self.assertTrue(scheduler._try_admit_prefill(runtime, now_ns))
        return runtime

    def test_remote_restore_path(self):
        scheduler = _make_scheduler()
        manager = scheduler.kv_manager
        _kv_admit_prefill(manager, "sy", 0, 100, 100, "ry")
        manager.mark_complete("sy", 100, "ry")
        # 唯一逐出路径:完整 session 整体外迁 -> REMOTE(模拟冷会话被动
        # 逐出后的下一 turn)。
        manager._evict_session(
            manager._sessions["sy"], now_ns=110, phase="admission",
            reason="probe", trigger_request_id="probe")
        snapshot = manager.session_snapshot("sy")
        self.assertEqual(snapshot.location, "remote_memory")
        self.assertIsNone(snapshot.instance_index)
        self.assertEqual(snapshot.resident_prefix_layers, 0)
        self.assertEqual(snapshot.local_bytes, 0)
        self.assertGreater(snapshot.remote_bytes, 0)

        runtime = self._admit_turn(
            scheduler, "ry1", "sy", 1,
            history_tokens=100, prefill_length=20,
            target_instance=0, now_ns=120)
        # REMOTE 全量回迁:KVTransfer = 单笔 remote_load [0, L),
        # reason = history_remote_restore(与决策同条承载)。
        self.assertEqual(runtime.history_action, "REMOTE_RESTORE")
        self.assertEqual(
            [(t.kind, t.reason) for t in runtime.history_transfers],
            [("remote_load", "history_remote_restore")])
        transfer = runtime.history_transfers[0]
        self.assertEqual(
            (transfer.layer_start, transfer.layer_end),
            (0, MODEL.layers))
        # remaining_chunks = ceil(20/128) = 1(recompute 项已删除,字段随
        # 2026-09-26 恒零管道清理移除)。
        self.assertEqual(runtime.remaining_chunks, 1)
        restored = manager.session_snapshot("sy")
        self.assertEqual(restored.location, "local_hbm")
        self.assertEqual(restored.resident_prefix_layers, MODEL.layers)
        # 恢复后完整本地驻留:全量字节在本地、远端零残留(_admit_turn 返回
        # 时点已含 grow_prefill 至 prefill_context=120 token)。
        self.assertEqual(
            restored.local_bytes, _shard_per_rank(120) * TP)
        self.assertEqual(restored.local_bytes, restored.total_bytes)
        self.assertEqual(restored.remote_bytes, 0)
        self.assertTrue(restored.active)

    def test_remote_restore_defers_while_store_unemitted(self):
        """B4 决策→发射窗口补丁(2026-10-01):decode 侧逐出"已决策未发射"
        的窗口内,REMOTE 会话下一 turn 的回迁准入被推迟(纯推迟,无
        mutation);store 发射(计数核销)后同一请求放行。防池读先于
        池写。"""
        scheduler = _make_scheduler()
        manager = scheduler.kv_manager
        _kv_admit_prefill(manager, "sy", 0, 100, 100, "ry")
        manager.mark_complete("sy", 100, "ry")
        manager._evict_session(
            manager._sessions["sy"], now_ns=110, phase="decode",
            reason="decode_target_capacity", trigger_request_id="probe")
        # 模拟"已决策未发射"窗口:graph 桩报告存在 pending store 决策。
        pending = {"sy": True}
        scheduler.graph.has_pending_store_decision = (
            lambda session_id: pending.get(session_id, False))
        runtime = _make_runtime(
            "ry1", "sy", turn=1,
            prefill_context_tokens=120, final_context_tokens=120,
            prefill_length=20, history_tokens_before=100)
        runtime.prefill_instance_index = 0
        self.assertFalse(scheduler._try_admit_prefill(runtime, 120))
        # 推迟是纯推迟:会话仍 REMOTE、未发生任何准入 mutation。
        self.assertIn(0, scheduler.prefill_admission_dirty)
        snapshot = manager.session_snapshot("sy")
        self.assertEqual(snapshot.location, "remote_memory")
        self.assertFalse(runtime.admitted_prefill)
        # store 发射(计数核销)后:同一决策边界之后的重试放行。
        pending["sy"] = False
        runtime.prefill_attempt_epoch = -1  # 强制重开纪元门(dirty 亦可)
        self.assertTrue(scheduler._try_admit_prefill(runtime, 121))
        self.assertEqual(runtime.history_action, "REMOTE_RESTORE")


def _verify_ready_scheduler(scheduler):
    """方案一(2026-10-20)测试辅助:在 __new__ 夹具上补齐 verify_run_end
    全部前置断言所需的账本(健康空跑终态:零交付/零在飞/零排队/实例
    空闲/graph 收尾表全空),使收尾观测段可直达。返回同一调度器。"""
    scheduler.runtime_by_request_id = {}
    scheduler._runtime_index = {}
    scheduler.runtimes = []
    scheduler.completed_requests = 0
    scheduler.expected_request_count = 0
    scheduler.arrival_heap = []
    scheduler._pending_first_steps = {}
    scheduler.graph = SimpleNamespace(
        sync_pending_history_after_evictions=lambda transfers: None,
        emit_eviction_actions=lambda plan, transfers: None,
        has_pending_store_decision=lambda session_id: False,
        note_request_complete=(
            lambda session_id, following, location: None),
        retire_request_state=lambda request_id: None,
        retire_completion_gate=lambda session_id: None,
        pending_store_tails={}, pending_store_decisions={},
        completion_gates={})
    scheduler.ack_count = 0
    scheduler.delivery_count = 0
    scheduler.last_applied_sequence = -1
    scheduler._delivery_reply_cache = None
    scheduler.in_flight = set()
    scheduler.arrived_request_count = 0
    scheduler.completed_request_count = 0
    scheduler.unseen_request_ids = set()
    scheduler._emitted_by_delivery = {}
    scheduler._acked_through = -1
    scheduler._acked_out_of_order = set()
    scheduler._provisional_kv_actions = {}
    scheduler.sensing_enabled = False
    return scheduler


class AdmissionBlockedEnrichmentTest(_PChunkHolderReset, unittest.TestCase):
    """方案一(2026-10-20)风险一:admission_blocked 决策行增富——两 stage
    (prefill/decode)× 两形态(无会话全量;LOCAL_HBM 同实例增量)的
    字段与数值;streak 三表记账与进展清账(max 保留)。数值全部由
    kv_cache_shard_bytes_for_tokens / workspace_fit_remaining_bytes /
    session_snapshot 在测试内独立复算(阻塞行只读口径的红线校验)。"""

    _RANKS = (0, 1)  # 实例 0 的 rank 序(topology.instances[0].ranks)

    def _blocked_rows(self, scheduler):
        return [row for row in scheduler.online_log_rows
                if row["kind"] == "admission_blocked"]

    def _assert_row_shape(self, row, request_id, stage, instance_index,
                          tick, expected_free, expected_required):
        self.assertEqual(row["request_id"], request_id)
        self.assertEqual(row["tick"], tick)
        self.assertEqual(row["priority"], 0)
        self.assertEqual(row["kind"], "admission_blocked")
        decision = row["decision"]
        self.assertEqual(set(decision), {
            "stage", "instance_index", "capacity_epoch",
            "kv_free_bytes_by_rank", "required_bytes_by_rank"})
        self.assertEqual(decision["stage"], stage)
        self.assertEqual(decision["instance_index"], instance_index)
        self.assertEqual(
            decision["kv_free_bytes_by_rank"], expected_free)
        self.assertEqual(
            decision["required_bytes_by_rank"], expected_required)

    def test_prefill_stage_full_requirement_row(self):
        scheduler = _make_scheduler()
        manager = scheduler.kv_manager
        avail = _occupancy_bytes(manager)
        # 无会话 + ACTIVE sy 占满目标实例(超量需求会被管理器对空实例
        # fail-closed 拒绝,故用"空实例可容纳、当前不可容纳"形态):
        # required = 全量分片(无增量信用)。
        tokens_p = _tokens_for_bytes(int(avail * 0.6))
        tokens_y = _tokens_for_bytes(
            avail - _shard_per_rank(tokens_p) + 1)
        _kv_admit_prefill(manager, "sy", 1, tokens_y, 100, "ry")
        expected_free = {
            rank: manager.workspace_fit_remaining_bytes(rank)
            for rank in (2, 3)}  # 实例 1 的 rank 序
        expected_required = dict(zip(
            (2, 3),
            kv_cache_shard_bytes_for_tokens(MODEL, tokens_p, TP)))
        runtime = _make_runtime("rx", "sx", prefill_context_tokens=tokens_p)
        runtime.prefill_instance_index = 1
        self.assertFalse(scheduler._try_admit_prefill(runtime, 1000))
        rows = self._blocked_rows(scheduler)
        self.assertEqual(len(rows), 1)
        self._assert_row_shape(rows[0], "rx", "prefill", 1, 1000,
                               expected_free, expected_required)
        self.assertEqual(scheduler._admission_blocked_streak, {"rx": 1})
        self.assertEqual(scheduler._admission_blocked_streak_max, {"rx": 1})
        self.assertEqual(
            scheduler._admission_blocked_open["rx"]["stage"], "prefill")

    def test_prefill_stage_incremental_requirement_row(self):
        scheduler = _make_scheduler()
        manager = scheduler.kv_manager
        avail = _occupancy_bytes(manager)
        # 会话本地驻留(inactive)在目标实例 0 上:required = 增量
        # (prefill_context 分片 − 已驻留 local_shard_bytes)。
        tokens_s = _tokens_for_bytes(int(avail * 0.4))
        _kv_admit_prefill(manager, "sx", 0, tokens_s, 100, "rx")
        manager.mark_complete("sx", 110, "rx")
        tokens_y = _tokens_for_bytes(
            int((avail - _shard_per_rank(tokens_s)) * 0.9))
        _kv_admit_prefill(manager, "sy", 0, tokens_y, 120, "ry")
        tokens_f = _tokens_for_bytes(avail + 100)
        snapshot = manager.session_snapshot("sx")
        self.assertEqual(snapshot.location, LOCAL_HBM)
        self.assertEqual(snapshot.instance_index, 0)
        full_shards = kv_cache_shard_bytes_for_tokens(MODEL, tokens_f, TP)
        expected_required = {
            rank: want - have for rank, want, have in zip(
                self._RANKS, full_shards, snapshot.local_shard_bytes)}
        expected_free = {
            rank: manager.workspace_fit_remaining_bytes(rank)
            for rank in self._RANKS}
        runtime = _make_runtime(
            "rx1", "sx", turn=1, prefill_context_tokens=tokens_f,
            final_context_tokens=tokens_f, prefill_length=tokens_f - tokens_s,
            history_tokens_before=tokens_s)
        runtime.prefill_instance_index = 0
        self.assertFalse(scheduler._try_admit_prefill(runtime, 200))
        rows = self._blocked_rows(scheduler)
        self.assertEqual(len(rows), 1)
        self._assert_row_shape(rows[0], "rx1", "prefill", 0, 200,
                               expected_free, expected_required)

    def test_decode_stage_full_requirement_row(self):
        scheduler = _make_scheduler()
        manager = scheduler.kv_manager
        avail = _occupancy_bytes(manager)
        # 会话 ACTIVE 驻留实例 0,decode target = 实例 1(ACTIVE sy 占
        # 满):required = 全量(双驻留口径,不给源副本信用)。
        tokens_x = _tokens_for_bytes(int(avail * 0.6))
        tokens_y = _tokens_for_bytes(
            avail - _shard_per_rank(tokens_x) + 1)
        _kv_admit_prefill(manager, "sx", 0, tokens_x, 100, "rx")
        _kv_admit_prefill(manager, "sy", 1, tokens_y, 100, "ry")
        expected_free = {
            rank: manager.workspace_fit_remaining_bytes(rank)
            for rank in (2, 3)}  # 实例 1 的 rank 序
        expected_required = dict(zip(
            (2, 3),
            kv_cache_shard_bytes_for_tokens(MODEL, tokens_x, TP)))
        runtime = _make_runtime(
            "rx", "sx", prefill_context_tokens=tokens_x,
            final_context_tokens=tokens_x)
        runtime.prefill_instance_index = 0
        runtime.decode_instance_index = 1
        runtime.waiting_decode_admission = True
        scheduler.waiting_decode_admissions[1].append(runtime)
        scheduler._try_admit_waiting_decodes(1000)
        rows = self._blocked_rows(scheduler)
        self.assertEqual(len(rows), 1)
        self._assert_row_shape(rows[0], "rx", "decode", 1, 1000,
                               expected_free, expected_required)

    def test_decode_stage_incremental_requirement_row(self):
        scheduler = _make_scheduler()
        manager = scheduler.kv_manager
        avail = _occupancy_bytes(manager)
        # 会话 ACTIVE 驻留实例 0 且 decode target = 实例 0(同实例):
        # required = 增量(final 分片 − 已驻留 local_shard_bytes)。
        tokens_s = _tokens_for_bytes(int(avail * 0.4))
        _kv_admit_prefill(manager, "sx", 0, tokens_s, 100, "rx")
        tokens_y = _tokens_for_bytes(
            int((avail - _shard_per_rank(tokens_s)) * 0.9))
        _kv_admit_prefill(manager, "sy", 0, tokens_y, 100, "ry")
        tokens_f = _tokens_for_bytes(avail + 100)
        snapshot = manager.session_snapshot("sx")
        full_shards = kv_cache_shard_bytes_for_tokens(MODEL, tokens_f, TP)
        expected_required = {
            rank: want - have for rank, want, have in zip(
                self._RANKS, full_shards, snapshot.local_shard_bytes)}
        expected_free = {
            rank: manager.workspace_fit_remaining_bytes(rank)
            for rank in self._RANKS}
        runtime = _make_runtime(
            "rx", "sx", prefill_context_tokens=tokens_s,
            final_context_tokens=tokens_f)
        runtime.prefill_instance_index = 0
        runtime.decode_instance_index = 0
        runtime.waiting_decode_admission = True
        scheduler.waiting_decode_admissions[0].append(runtime)
        scheduler._try_admit_waiting_decodes(1000)
        rows = self._blocked_rows(scheduler)
        self.assertEqual(len(rows), 1)
        self._assert_row_shape(rows[0], "rx", "decode", 0, 1000,
                               expected_free, expected_required)

    def test_streak_tables_progress_clears_current_keeps_max(self):
        scheduler = _make_scheduler()
        manager = scheduler.kv_manager
        avail = _occupancy_bytes(manager)
        tokens_p = _tokens_for_bytes(int(avail * 0.6))
        tokens_y = _tokens_for_bytes(
            avail - _shard_per_rank(tokens_p) + 1)
        _kv_admit_prefill(manager, "sy", 1, tokens_y, 100, "ry")
        runtime = _make_runtime("rx", "sx", prefill_context_tokens=tokens_p)
        runtime.prefill_instance_index = 1
        scheduler._try_admit_prefill(runtime, 1000)
        scheduler._try_admit_prefill(runtime, 1001)  # dirty 门重试,再阻塞
        self.assertEqual(scheduler._admission_blocked_streak, {"rx": 2})
        self.assertEqual(scheduler._admission_blocked_streak_max, {"rx": 2})
        # 进展清账(_try_admit_prefill 成功路径):当前 streak 与 open
        # 登记弹出,max 保留(供 verify_run_end 直方图)。
        scheduler._clear_admission_blocked_progress("rx")
        self.assertEqual(scheduler._admission_blocked_streak, {})
        self.assertEqual(scheduler._admission_blocked_open, {})
        self.assertEqual(scheduler._admission_blocked_streak_max, {"rx": 2})
        # 再阻塞再清:max 单调不回落。
        scheduler._try_admit_prefill(runtime, 1002)
        self.assertEqual(scheduler._admission_blocked_streak, {"rx": 1})
        scheduler._clear_admission_blocked_progress("rx")
        self.assertEqual(scheduler._admission_blocked_streak_max, {"rx": 2})


class AdmissionBlockedVerifyEndTest(_PChunkHolderReset, unittest.TestCase):
    """方案一(2026-10-20)风险一:verify_run_end 增补——open 登记残留
    即 raise(fail-closed);请求级 max streak 直方图结算。"""

    def test_open_registry_leak_raises(self):
        scheduler = _verify_ready_scheduler(_make_scheduler())
        # 模拟进展清账链断裂:请求已不在任何排队容器(健康终态),但
        # open 登记残留 -> 收尾观测段显影即 raise。
        scheduler._admission_blocked_open = {
            "rx": {"stage": "decode", "instance_index": 1,
                   "tick": 900, "streak": 2}}
        with self.assertRaises(RuntimeError) as raised:
            scheduler.verify_run_end()
        self.assertIn("still admission-blocked without progress",
                      str(raised.exception))
        self.assertIn("rx", str(raised.exception))

    def test_histogram_settled_from_max_table(self):
        scheduler = _verify_ready_scheduler(_make_scheduler())
        scheduler._admission_blocked_streak_max = {"ra": 2, "rb": 2, "rc": 5}
        scheduler.verify_run_end()
        self.assertEqual(
            scheduler.admission_blocked_streak_histogram, {"2": 2, "5": 1})
        # 健康零阻塞 run:空直方图(而非 None——结算已发生)。
        clean = _verify_ready_scheduler(_make_scheduler())
        clean.verify_run_end()
        self.assertEqual(clean.admission_blocked_streak_histogram, {})


class WorkspaceFloorReconciliationTest(_PChunkHolderReset,
                                       unittest.TestCase):
    """方案一(2026-10-20)风险三:workspace 峰值采集(逐 rank 运行最大
    值,回落不缩)+ verify_run_end 地板对账(slack 结算 / 超限 raise /
    非 required 保持 None);外部视图回落信号(epoch 推进)一并钉测
    (审计表 #1)。"""

    _FLOOR = 5000

    def _required_scheduler(self, prospective=None):
        from session_kv_manager import SessionKVCacheManager
        scheduler = _verify_ready_scheduler(_make_scheduler())
        topology = scheduler.topology
        floor = (prospective if prospective is not None else {
            rank: self._FLOOR
            for instance in topology.instances
            for rank in instance.ranks})
        scheduler.kv_manager = SessionKVCacheManager(
            topology, MODEL, prospective_workspace_bytes_by_rank=floor)
        scheduler.workspace_admission = "required"
        scheduler._workspace_rank_to_instance = {
            rank: instance.index
            for instance in topology.instances
            for rank in instance.ranks}
        return scheduler, sorted(floor)

    @staticmethod
    def _delta(sequence, active_by_rank, ranks):
        return {
            "schema_version": 2, "delivery_sequence": sequence,
            "delivery_epoch": sequence, "tick": sequence,
            "deferred_from_tick": sequence, "reasons": ["TICK"],
            "arrivals": [], "completed_groups": [], "completed_nodes": [],
            "affected_ranks": [],
            "snapshot_handle": {"epoch": sequence, "tick": sequence},
            "workspace_state": {
                "protocol_version": 1,
                "ranks": [
                    {"rank": rank, "workspace_revision": sequence + 1,
                     "active_workspace_bytes": active_by_rank[rank]}
                    for rank in ranks]}}

    def test_peak_tracking_and_slack_reconciliation(self):
        scheduler, ranks = self._required_scheduler()
        scheduler._apply_delivery_extension(
            self._delta(0, {rank: 3000 for rank in ranks}, ranks))
        epoch_after_first = list(scheduler.capacity_epoch)
        # 回落交付:峰值保持运行最大值(不随回落缩),视图变化仍推进
        # 所属实例纪元(审计表 #1:外部 workspace 视图回落必有信号)。
        scheduler._apply_delivery_extension(
            self._delta(1, {rank: 1200 for rank in ranks}, ranks))
        self.assertEqual(
            scheduler._workspace_peak_bytes_by_rank,
            {rank: 3000 for rank in ranks})
        self.assertEqual(
            scheduler.capacity_epoch,
            [value + 1 for value in epoch_after_first])
        scheduler.verify_run_end()
        self.assertEqual(scheduler.workspace_floor_reconciliation, {
            "peak_bytes_by_rank": {str(rank): 3000 for rank in ranks},
            "floor_bytes_by_rank": {
                str(rank): self._FLOOR for rank in ranks},
            "slack_bytes_by_rank": {str(rank): 2000 for rank in ranks},
        })

    def test_peak_above_floor_raises(self):
        scheduler, ranks = self._required_scheduler()
        scheduler._apply_delivery_extension(
            self._delta(0, {rank: self._FLOOR + 1 for rank in ranks}, ranks))
        with self.assertRaises(RuntimeError) as raised:
            scheduler.verify_run_end()
        self.assertIn(
            "workspace peak exceeded prospective certificate floor",
            str(raised.exception))
        self.assertIn("rank {}".format(ranks[0]), str(raised.exception))

    def test_not_required_stays_none(self):
        # 非 required 模式:无证书即无对账对象,公开属性保持 None;
        # _apply_delivery_extension 早退不采集峰值。
        scheduler = _verify_ready_scheduler(_make_scheduler())
        scheduler.verify_run_end()
        self.assertIsNone(scheduler.workspace_floor_reconciliation)
        self.assertEqual(scheduler._workspace_peak_bytes_by_rank, {})


class CapacityReleaseSignalTest(_PChunkHolderReset, unittest.TestCase):
    """方案一(2026-10-20)信号触点审计表钉测:每条容量释放路径后必
    有重试信号(epoch note 或 dirty 门)。审计结论零缺口,本组把
    结论钉进回归(逐出完成 / terminal retirement / REMOTE store 推迟)。
    外部视图回落信号在 WorkspaceFloorReconciliationTest;阻塞-逐出-
    重试-准入(drain 准入族)由本组逐出用例与既有用例 A/C 共同覆盖。"""

    def test_blocked_attempt_with_eviction_notes_capacity_then_readmits(
            self):
        """审计表 #5/#7:阻塞 attempt 的逐出是真实 mutation——即使最终
        阻塞,逐出 victim 的容量释放也必须推进纪元(否则纪元门对本
        实例永久关闭);释放后小请求重试即准入。"""
        scheduler = _make_scheduler()
        manager = scheduler.kv_manager
        avail = _occupancy_bytes(manager)
        # 实例 1:冷会话 sy(完成后可逐出,占 ~0.5)+ ACTIVE sz(占
        # ~0.3);需求 ~0.9(空实例可容纳,不触管理器 fail-closed)——
        # 逐出 sy 后仍不足 -> 带 evictions 的阻塞分支。
        tokens_sy = _tokens_for_bytes(int(avail * 0.5))
        tokens_sz = _tokens_for_bytes(int(avail * 0.3))
        _kv_admit_prefill(manager, "sy", 1, tokens_sy, 100, "ry")
        manager.mark_complete("sy", 110, "ry")
        _kv_admit_prefill(manager, "sz", 1, tokens_sz, 120, "rz0")
        tokens_p = _tokens_for_bytes(int(avail * 0.9))
        runtime = _make_runtime("rx", "sx", prefill_context_tokens=tokens_p)
        runtime.prefill_instance_index = 1
        self.assertFalse(scheduler._try_admit_prefill(runtime, 200))
        # 逐出已发生(真实 mutation,REMOTE 化)且纪元被推进。
        evicted = manager.session_snapshot("sy")
        self.assertEqual(evicted.location, "remote_memory")
        self.assertGreater(scheduler.capacity_epoch[1], 0)
        self.assertIn(1, scheduler.prefill_admission_dirty)
        # 阻塞行在场(增富);释放后的重试对小请求放行(drain 准入)。
        self.assertTrue(any(row["kind"] == "admission_blocked"
                            for row in scheduler.online_log_rows))
        small = _make_runtime(
            "rw", "sw", prefill_context_tokens=_tokens_for_bytes(1))
        small.prefill_instance_index = 1
        self.assertTrue(scheduler._try_admit_prefill(small, 300))

    def test_terminal_retirement_notes_capacity(self):
        """审计表 #3:terminal 会话回收释放本地 HBM(含 reservation 经
        内部 release_request_capacity 的释放)——released 实例纪元必须
        推进,字节实减可复核。"""
        scheduler = _make_scheduler()
        manager = scheduler.kv_manager
        tokens = _tokens_for_bytes(int(_occupancy_bytes(manager) * 0.3))
        _kv_admit_prefill(manager, "sx", 0, tokens, 100, "rx")
        manager.mark_complete("sx", 110, "rx")
        before_free = manager.hbm_snapshots(0)[0].remaining_bytes
        runtime = _make_runtime(
            "rx", "sx", prefill_context_tokens=tokens,
            final_context_tokens=tokens)
        runtime.kv_state_after_completion = LOCAL_HBM
        scheduler.runtime_by_request_id = {"rx": runtime}
        scheduler._runtime_index = {"rx": 0}
        scheduler.runtimes = [runtime]
        scheduler.next_request = [None]
        scheduler.completed_requests = 1
        scheduler._on_request_complete("rx", 200)
        self.assertIsNone(manager.session_snapshot("sx"))
        self.assertGreater(
            manager.hbm_snapshots(0)[0].remaining_bytes, before_free)
        self.assertEqual(scheduler.capacity_epoch[0], 1)
        self.assertNotIn("rx", scheduler.runtime_by_request_id)

    def test_remote_store_postpone_dirty_without_blocked_row(self):
        """审计表 #9(FL 特有):REMOTE store 推迟是纯推迟(零 mutation),
        信号 = dirty 门(store 发射后必然重试);该分支不属于容量阻塞,
        不产 admission_blocked 行(增富只覆盖两容量阻塞分支)。"""
        scheduler = _make_scheduler()
        manager = scheduler.kv_manager
        _kv_admit_prefill(manager, "sy", 0, 100, 100, "ry")
        manager.mark_complete("sy", 100, "ry")
        manager._evict_session(
            manager._sessions["sy"], now_ns=110, phase="decode",
            reason="decode_target_capacity", trigger_request_id="probe")
        scheduler.graph.has_pending_store_decision = (
            lambda session_id: True)
        runtime = _make_runtime(
            "ry1", "sy", turn=1,
            prefill_context_tokens=120, final_context_tokens=120,
            prefill_length=20, history_tokens_before=100)
        runtime.prefill_instance_index = 0
        self.assertFalse(scheduler._try_admit_prefill(runtime, 120))
        self.assertIn(0, scheduler.prefill_admission_dirty)
        self.assertEqual(scheduler._admission_blocked_streak, {})
        self.assertEqual(scheduler._admission_blocked_open, {})
        self.assertFalse(any(row["kind"] == "admission_blocked"
                             for row in scheduler.online_log_rows))


class AdmissionLivenessAuditToolTest(unittest.TestCase):
    """方案一(2026-10-20)风险一:离线活性审计器四态单测(PASS 无阻塞
    行 / PASS 短 streak 或不可行 / FAIL 连续可行 / SKIP 字段缺失)。
    审计器为仓内既有工具(主 agent 预放,四仓同 md5),此处只测行为。
    """

    @staticmethod
    def _row(seq, kind, request_id, decision=None):
        return {"seq": seq, "tick": seq * 10, "priority": 0,
                "kind": kind, "request_id": request_id,
                "decision": decision if decision is not None else {}}

    @staticmethod
    def _blocked(seq, request_id, free, required):
        return AdmissionLivenessAuditToolTest._row(
            seq, "admission_blocked", request_id,
            {"stage": "decode", "instance_index": 0, "capacity_epoch": 0,
             "kv_free_bytes_by_rank": {"0": free},
             "required_bytes_by_rank": {"0": required}})

    def _audit(self, rows):
        import tempfile
        from pathlib import Path
        from online.verify.admission_liveness_audit import main
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "online_decision_log.jsonl"
            with path.open("w", encoding="utf-8") as sink:
                for row in rows:
                    sink.write(json.dumps(row) + "\n")
            return main([str(path)])

    def test_pass_no_blocked_rows(self):
        rows = [self._row(1, "prefill", "ra"),
                self._row(2, "decode", "ra"),
                self._row(3, "completion", "ra")]
        self.assertEqual(self._audit(rows), 0)

    def test_pass_short_or_infeasible_streak(self):
        # 不可行阻塞(free < required)再长也不违例:容量确实不够。
        infeasible = [self._blocked(i, "ra", 100, 200) for i in range(1, 5)]
        self.assertEqual(self._audit(infeasible), 0)
        # 短 streak(2 < 缺省 3)即使可行也不判 FAIL。
        short = [self._blocked(1, "rb", 300, 200),
                 self._blocked(2, "rb", 300, 200)]
        self.assertEqual(self._audit(short), 0)
        # 进展行(prefill/decode/completion)切断计数:2+2 被切断不连续。
        cut = [self._blocked(1, "rc", 300, 200),
               self._blocked(2, "rc", 300, 200),
               self._row(3, "decode", "rc"),
               self._blocked(4, "rc", 300, 200),
               self._blocked(5, "rc", 300, 200)]
        self.assertEqual(self._audit(cut), 0)

    def test_fail_consecutive_feasible_streak(self):
        rows = [self._blocked(1, "ra", 300, 200),
                self._blocked(2, "ra", 300, 200),
                self._blocked(3, "ra", 300, 200)]
        self.assertEqual(self._audit(rows), 1)

    def test_skip_missing_enrichment_fields(self):
        # 旧构建的阻塞行无增富字段:明确跳过(不误判),退出码 0。
        rows = [self._row(i, "admission_blocked", "ra")
                for i in range(1, 4)]
        self.assertEqual(self._audit(rows), 0)


if __name__ == "__main__":
    unittest.main()
