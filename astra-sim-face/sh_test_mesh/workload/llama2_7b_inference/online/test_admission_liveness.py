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
        event for event in manager.events
        if event.event_type in ("admission_blocked", "admission_retry")
    ]


class DecodeAdmissionLivenessTest(unittest.TestCase):
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


class PrefillAdmissionLivenessTest(unittest.TestCase):
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


class DecodeSelectionFilterTest(unittest.TestCase):
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


class PrefillSelectionFilterTest(unittest.TestCase):
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


class ProbeSemanticsTest(unittest.TestCase):
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


class DecodeEvictionSerializationTest(unittest.TestCase):
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
        # 先序列化后置空语义;:1171 M4 置空保留为幂等兜底)。
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


if __name__ == "__main__":
    unittest.main()
