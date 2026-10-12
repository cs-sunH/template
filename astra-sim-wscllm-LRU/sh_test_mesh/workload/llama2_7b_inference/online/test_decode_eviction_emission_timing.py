#!/usr/bin/env python3
"""test_decode_eviction_emission_timing.py -- C-2/M-25 修复回归(2026-09-30)
:#2 预占逐出的"决策时点就近发射"时序钉子(调度器级)。

缺陷(graph_batch_builder 旧路径):#2 预占逐出在 prefill 准入时决定、随
joiner 首列车发射——决策到发射之间隔着逐出者整个 prefill 时长,窗口内
  (a) 受害会话下一轮回迁查不到 pending_store_tails,store→restore 前递
      边(策略 §4.4 唯一新增正确性边)丢失(倒序发射);
  (b) 已 terminal 会话的 store 支链退休后仍物理发射(死亡 NoC/远端池
      流量)且登记条目永久泄漏,run 末 fail-closed 把整跑作废。
修复:预占逐出登 decode_evictions_pending 子账本,准入成功随 prefill 批
发射(段块末同门)、阻塞 attempt 本边界立即发射(无门);#4/#5 move/
grow 逐出(运行时经预约不变量恒零)保留 joiner 列车头发射路径。

场景:
  1. 准入成功:_try_admit_prefill 只登账不发射;_emit_prefill 同边界
     gated=True 发射并核销待发射账本(决策账本 decode_target_evictions
     保留全集,decode 决策行序列化口径不变);
  2. 预占阻塞(全量 attempt 携逐出后仍不足):本边界 gated=False 立即
     发射,窗口不跨 capacity_epoch 重试存续;
  3. 列车发射:joiner 计划只携带待发射子集(#4/#5),发射后随 decode
     决策核销一并清空。

Run: python3 online/test_decode_eviction_emission_timing.py
"""
import os
import sys
import unittest
from types import SimpleNamespace

_ONLINE_DIR = os.path.dirname(os.path.abspath(__file__))
_WORKLOAD_DIR = os.path.dirname(_ONLINE_DIR)
for _p in (_ONLINE_DIR, _WORKLOAD_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from online.wsc_llm_online_scheduler import (  # noqa: E402
    _OnlineInstanceState,
    _OnlineRequestRuntime,
    WscLlmOnlineScheduler,
)
from session_kv_manager import (  # noqa: E402
    CapacityResult,
    HistoryDecision,
    KVTransfer,
)
from wsc_llm_scheduler import (  # noqa: E402  (只读 import)
    DECODE_ROLE,
    PREFILL_ROLE,
    WscLlmModel,
)


def _eviction(victim_instance_index, trigger_request_id, session_suffix=""):
    # 整体逐出:remote_store KVTransfer(source_instance_index 即受害实例;
    # 层域 [0, L) 全层、本地驻留清零)。
    return KVTransfer(
        kind="remote_store",
        phase="history",
        reason="static_decode_final_kv_reservation_session",
        session_id=f"victim_session_{victim_instance_index}{session_suffix}",
        trigger_request_id=trigger_request_id,
        source_instance_index=victim_instance_index,
        target_instance_index=None,
        total_bytes=0,
        shards=(),
        model_layers=32,
        layer_start=0,
        layer_end=32,
        resident_prefix_layers_before=32,
        resident_prefix_layers_after=0,
    )


class _StubKVManager:
    """桩 kv_manager:按脚本回放 reservation / history / growth 决策
    (完整 SessionKVCacheManager 构造过重,test_admission_eviction_
    accumulation 同法)。"""

    def __init__(self, reservations, history_decisions):
        self._reservations = list(reservations)
        self._history = list(history_decisions)
        self.calls = []

    def reserve_request_capacity(self, *args, **kwargs):
        self.calls.append(("reserve_request_capacity", args, kwargs))
        return self._reservations.pop(0)

    def release_request_capacity(self, *args, **kwargs):
        self.calls.append(("release_request_capacity", args, kwargs))

    def session_snapshot(self, session_id):
        return None

    def prepare_history(self, *args, **kwargs):
        self.calls.append(("prepare_history", args, kwargs))
        return self._history.pop(0)

    def grow_prefill(self, *args, **kwargs):
        self.calls.append(("grow_prefill", args, kwargs))
        return CapacityResult((), True, ())


class _RecordingGraph:
    """构图器桩:记录 emit_decode_eviction_actions 的 (plan, transfers,
    gated) 调用;emit_prefill_batch/emit_iteration_train 返回最小形状。"""

    def __init__(self):
        self.decode_eviction_calls = []
        self.prefill_batches = 0

    def sync_pending_history_after_evictions(self, transfers):
        pass

    def emit_decode_eviction_actions(self, plan, transfers, *,
                                     gated_on_prefill_segment_end=False):
        self.decode_eviction_calls.append(
            (plan, tuple(transfers), gated_on_prefill_segment_end))

    def emit_prefill_batch(self, plan):
        self.prefill_batches += 1
        return {}

    def emit_iteration_train(self, plan):
        return {
            "exit_members": {
                member["request_id"]: [member["request_id"]]
                for member in plan["exit_members"]},
            "sentinel_members": {}}


def _admission_runtime():
    record = {
        "request_id": "session_9_request_0",
        "session_id": "session_9",
        "turn_index": 0,
        "queue_index": 0,
        "prefill_length": 512,
        "decode_length": 4,
        "history_tokens_before": 0,
        "prefill_context_tokens": 512,
        "final_context_tokens": 516,
    }
    runtime = _OnlineRequestRuntime(record)
    runtime.prefill_instance_index = 1
    runtime.prefill_assignment_key = [0]
    runtime.static_route = SimpleNamespace(
        prefill_instance_index=1,
        decode_instance_index=0,
        path=(1, 0),
        hop_count=1,
        shared_edges=())
    return runtime


def _admission_scheduler(kv_manager, graph):
    # __new__ 跳过 __init__,只补 _try_admit_prefill / _emit_prefill 触达
    # 的字段(test_admission_eviction_accumulation 同法)。
    scheduler = WscLlmOnlineScheduler.__new__(WscLlmOnlineScheduler)
    scheduler.instances = [
        _OnlineInstanceState(index=0, phase_role=DECODE_ROLE),
        _OnlineInstanceState(index=1, phase_role=PREFILL_ROLE),
    ]
    scheduler.runtime_by_request_id = {}
    scheduler.capacity_epoch = [0, 0]
    scheduler.kv_manager = kv_manager
    scheduler.graph = graph
    scheduler.config = SimpleNamespace(
        model=WscLlmModel(
            layers=32, hidden_size=4096, ffn_size=11008, num_heads=32,
            vocab_size=32000, bytes_per_elem=2),
        prefill_chunk_size=128)
    scheduler.topology = SimpleNamespace(
        instances=[SimpleNamespace(size=1)])
    # _emit_prefill 驱动补充装配(test_decode_eviction_serialization 同法)。
    scheduler._emitted_by_delivery = {1: {"requests": []}}
    scheduler._batch = {"delivery_sequence": 1, "watches": [],
                        "assignments": []}
    scheduler.ledger_issued = {}
    scheduler.train_ledger_sink = None
    scheduler.online_log_rows = []
    scheduler.online_log_count = 0
    scheduler.decision_log_sink = None
    return scheduler


class ReserveEvictionFlushTimingTest(unittest.TestCase):
    """1/2. #2 预占逐出的决策时点就近发射时序。"""

    def test_successful_admission_flushes_gated_at_prefill_emission(self):
        """准入成功:预占逐出随 prefill 批发射(段块末触发门),决策到
        发射窗口收敛为零;决策账本保留全集。"""
        eviction = _eviction(0, "session_9_request_0")
        kv_manager = _StubKVManager(
            reservations=[CapacityResult((eviction,), True, ())],
            history_decisions=[
                HistoryDecision(
                    action="NO_HISTORY", source_instance_index=None,
                    target_instance_index=1, history_tokens=0,
                    transfers=(), evictions=()),
            ],
        )
        graph = _RecordingGraph()
        scheduler = _admission_scheduler(kv_manager, graph)
        runtime = _admission_runtime()

        self.assertTrue(scheduler._try_admit_prefill(runtime, 1_000))
        # 决策时点只登账不发射(发射点 = prefill 批,见 _emit_prefill)。
        self.assertEqual(runtime.decode_evictions_pending, (eviction,))
        self.assertEqual(graph.decode_eviction_calls, [])

        scheduler._emit_prefill(runtime, 1_000)
        # 同一准入边界内随 prefill 批 gated 发射并核销待发射账本。
        self.assertEqual(len(graph.decode_eviction_calls), 1)
        plan, transfers, gated = graph.decode_eviction_calls[0]
        self.assertEqual(plan["request_id"], "session_9_request_0")
        self.assertEqual(transfers, (eviction,))
        self.assertTrue(gated)
        self.assertEqual(runtime.decode_evictions_pending, ())
        # 决策日志账本(decode 决策行序列化源)保留全集。
        self.assertEqual(runtime.decode_target_evictions, (eviction,))

    def test_prefill_ledger_row_chunks_are_positive_ceil(self):
        """M-26 回归(2026-09-30):P 侧台账行 prefill_chunks/iterations/
        pass_spans = ceil(prefill_length / chunk) 恒正(旧负 floor 表达式
        -(len // chunk) 让三字段恒负/零,违反自身 prefill_chunks>0 契约)。"""
        kv_manager = _StubKVManager(
            reservations=[CapacityResult((), True, ())],
            history_decisions=[
                HistoryDecision(
                    action="NO_HISTORY", source_instance_index=None,
                    target_instance_index=1, history_tokens=0,
                    transfers=(), evictions=()),
            ],
        )
        graph = _RecordingGraph()
        scheduler = _admission_scheduler(kv_manager, graph)
        ledger_rows = []
        scheduler.train_ledger_sink = ledger_rows.append
        runtime = _admission_runtime()

        self.assertTrue(scheduler._try_admit_prefill(runtime, 1_000))
        scheduler._emit_prefill(runtime, 1_000)
        self.assertEqual(len(ledger_rows), 1)
        row = ledger_rows[0]
        # 512 tokens / 128 chunk = 4(整除也必须取正;负 floor 会给 -4)。
        self.assertEqual(
            (row["prefill_chunks"], row["iterations"], row["pass_spans"]),
            (4, 4, 4))
        self.assertGreater(row["prefill_chunks"], 0)

    def test_blocked_attempt_flushes_ungated_immediately(self):
        """预占阻塞:本边界立即发射(无门),窗口不跨 capacity_epoch
        重试存续(阻塞期间受害会话回迁必见登记尾部)。

        口径注记(2026-09-30 审查):本用例的伪造 CapacityResult
        ((eviction,), False, ...) 在现管理器下结构性不可达(边际门保证
        blocked ⟹ evictions==()),锁定的是防御臂语义本身,非生产行为。"""
        eviction = _eviction(0, "session_9_request_0")
        kv_manager = _StubKVManager(
            reservations=[CapacityResult((eviction,), False, (0,))],
            history_decisions=[])
        graph = _RecordingGraph()
        scheduler = _admission_scheduler(kv_manager, graph)
        runtime = _admission_runtime()

        self.assertFalse(scheduler._try_admit_prefill(runtime, 1_000))
        self.assertEqual(len(graph.decode_eviction_calls), 1)
        plan, transfers, gated = graph.decode_eviction_calls[0]
        self.assertEqual(transfers, (eviction,))
        self.assertFalse(gated)
        self.assertEqual(runtime.decode_evictions_pending, ())
        self.assertEqual(runtime.decode_target_evictions, (eviction,))

    def test_flush_without_pending_evictions_is_noop(self):
        """无逐出准入(常态):发射入口零调用,不产生空发射段。"""
        kv_manager = _StubKVManager(
            reservations=[CapacityResult((), True, ())],
            history_decisions=[
                HistoryDecision(
                    action="NO_HISTORY", source_instance_index=None,
                    target_instance_index=1, history_tokens=0,
                    transfers=(), evictions=()),
            ],
        )
        graph = _RecordingGraph()
        scheduler = _admission_scheduler(kv_manager, graph)
        runtime = _admission_runtime()

        self.assertTrue(scheduler._try_admit_prefill(runtime, 1_000))
        scheduler._emit_prefill(runtime, 1_000)
        self.assertEqual(graph.decode_eviction_calls, [])
        self.assertEqual(runtime.decode_evictions_pending, ())


class TrainJoinPendingEvictionTest(unittest.TestCase):
    """3. joiner 列车头只承载待发射子集(#4/#5),发射后核销。"""

    def test_emit_train_carries_pending_then_writes_off(self):
        scheduler = WscLlmOnlineScheduler.__new__(WscLlmOnlineScheduler)
        scheduler.instances = [
            _OnlineInstanceState(index=0, phase_role=DECODE_ROLE),
        ]
        scheduler.runtime_by_request_id = {}
        scheduler.capacity_epoch = [0]
        scheduler.config = SimpleNamespace(
            model=WscLlmModel(
                layers=32, hidden_size=4096, ffn_size=11008, num_heads=32,
                vocab_size=32000, bytes_per_elem=2))
        scheduler.topology = SimpleNamespace(
            instances=[SimpleNamespace(size=1)])
        scheduler._train_max_iter = 0
        scheduler._train_instance_index = {}
        scheduler._ready_frontier = set()
        scheduler._emitted_by_delivery = {1: {"requests": []}}
        scheduler._batch = {"delivery_sequence": 1, "watches": []}
        scheduler.ledger_issued = {}
        scheduler.train_ledger_sink = None
        scheduler.online_log_rows = []
        scheduler.online_log_count = 0
        scheduler.decision_log_sink = None
        graph = _RecordingGraph()
        scheduler.graph = graph

        runtime = _admission_runtime()
        eviction = _eviction(0, "session_9_request_0")
        # #2 预占逐出已决策并发射(账本保留全集、待发射账本已被
        # _flush_decode_evictions 清空)——列车头不得重发。
        runtime.decode_target_evictions = (eviction,)
        runtime.decode_evictions_pending = ()
        runtime.waiting_decode_admission = False
        state = scheduler.instances[0]
        state.active_decode.append(runtime)
        state.active_decode_lookup.add(runtime)
        scheduler.runtime_by_request_id[runtime.request_id] = runtime

        scheduler._emit_train(state, tick=2_000)

        # 决策行仍序列化全集(发射时点快照口径不变)。
        rows = [row for row in scheduler.online_log_rows
                if row["kind"] == "decode"]
        self.assertEqual(len(rows), 1)
        self.assertEqual(
            rows[0]["decision"]["decode_target_evictions"],
            [{
                "kind": "remote_store",
                "reason": "static_decode_final_kv_reservation_session",
                "session_id": "victim_session_0",
                "total_bytes": 0,
                "source_instance_index": 0,
                "target_instance_index": None,
                "layer_start": 0,
                "layer_end": 32,
            }])
        self.assertEqual(runtime.decode_target_evictions, ())
        self.assertEqual(runtime.decode_evictions_pending, ())


if __name__ == "__main__":
    unittest.main()
