#!/usr/bin/env python3
"""test_admission_eviction_accumulation.py -- wscllm admission_evictions
跨 attempt 累积语义钉子(2026-09-05 修复回测)。

缺陷:wsc_llm_online_scheduler.py `_try_admit_prefill` 的
``runtime.admission_evictions = decision.evictions``(:1797 对应行)是
覆盖赋值——同函数内 decode_target(:1773-1775)与 growth(:1834)均为
追加。prepare_history 在 admission_blocked=True 时可能已真实逐出
(session_kv_manager.py:1577-1593);capacity_epoch 变化后重试成功时,
第一次 attempt 的逐出被覆盖丢失,decision log(:1412-1415 序列化整个
runtime.admission_evictions)将少记。修复:改为追加语义,与同函数
另两处对齐。

场景(方案 T4):第一次 _try_admit_prefill 因 prepare_history 返回
admission_blocked=True 且 evictions=(E1,) 拒绝 -> capacity_epoch 变化
后重试 -> 第二次 prepare_history 返回 evictions=(E2,) 放行 ->
断言 runtime.admission_evictions == (E1, E2)(修复前为 (E2,))。

Run: python3 online/test_admission_eviction_accumulation.py
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
    WscLlmModel,
)


def _eviction(victim_instance_index, trigger_request_id, time_ns=1_000):
    # B2 三态:逐出对象 = remote_store KVTransfer(source_instance_index 即
    # 受害实例,纪元唤醒与序列化消费同一字段;time_ns 不进对象)。
    return KVTransfer(
        kind="remote_store",
        phase="history",
        reason="watermark_admission_full_fallback",
        session_id=f"victim_session_{victim_instance_index}",
        trigger_request_id=trigger_request_id,
        source_instance_index=victim_instance_index,
        target_instance_index=None,
        total_bytes=0,
        shards=(),
        model_layers=1,
        layer_start=0,
        layer_end=1,
        resident_prefix_layers_before=1,
        resident_prefix_layers_after=0,
    )


class _StubKVManager:
    """桩 kv_manager:按脚本依次回放 reservation / history / growth 决策,
    记录调用序供断言(完整 SessionKVCacheManager 构造过重,方案 T4 允许)。"""

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

    def hbm_snapshots(self, instance_index):
        return None

    def prepare_history(self, *args, **kwargs):
        self.calls.append(("prepare_history", args, kwargs))
        return self._history.pop(0)

    def grow_prefill(self, *args, **kwargs):
        self.calls.append(("grow_prefill", args, kwargs))
        return CapacityResult((), True, ())


def _runtime():
    record = {
        "request_id": "session_0_request_0",
        "session_id": "session_0",
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
    runtime.static_route = SimpleNamespace(decode_instance_index=0)
    return runtime


def _scheduler(kv_manager):
    # _bare_scheduler 构造方式(test_train_machinery.py:71):__new__ 跳过
    # __init__,只补 _try_admit_prefill 触达的字段。
    scheduler = WscLlmOnlineScheduler.__new__(WscLlmOnlineScheduler)
    scheduler.instances = [
        _OnlineInstanceState(index=0, phase_role=DECODE_ROLE),
        _OnlineInstanceState(index=1, phase_role=DECODE_ROLE),
    ]
    scheduler.runtime_by_request_id = {}
    scheduler.capacity_epoch = [0, 0]
    scheduler.kv_manager = kv_manager
    # B3(2026-09-06):_try_admit_prefill 现场调用发射账本镜像
    # (sync_pending_history_after_evictions)——桩挂 no-op。
    scheduler.graph = SimpleNamespace(
        sync_pending_history_after_evictions=lambda transfers: None)
    scheduler.config = SimpleNamespace(
        model=WscLlmModel(
            layers=32, hidden_size=4096, ffn_size=11008, num_heads=32,
            vocab_size=32000, bytes_per_elem=2))
    scheduler.topology = SimpleNamespace(
        instances=[SimpleNamespace(size=1)])
    return scheduler


class AdmissionEvictionAccumulationTest(unittest.TestCase):
    def test_blocked_attempt_evictions_survive_retry(self):
        """第一次 attempt 的 history 逐出(E1)不得被放行的第二次 attempt
        (E2)覆盖:runtime.admission_evictions == (E1, E2)。"""
        e1 = _eviction(0, "session_0_request_0", time_ns=1_000)
        e2 = _eviction(0, "session_0_request_0", time_ns=2_000)
        kv_manager = _StubKVManager(
            reservations=[
                CapacityResult((), True, ()),   # attempt 1: 容量预占放行
                CapacityResult((), True, ()),   # attempt 2: 容量预占放行
            ],
            history_decisions=[
                # attempt 1: prepare_history 逐出 E1 后 admission_blocked
                HistoryDecision(
                    action="REMOTE_RESTORE", source_instance_index=1,
                    target_instance_index=1, history_tokens=0,
                    transfers=(),
                    evictions=(e1,), admission_blocked=True),
                # attempt 2: prepare_history 逐出 E2 后放行
                HistoryDecision(
                    action="REMOTE_RESTORE", source_instance_index=1,
                    target_instance_index=1, history_tokens=0,
                    transfers=(),
                    evictions=(e2,), admission_blocked=False),
            ],
        )
        scheduler = _scheduler(kv_manager)
        runtime = _runtime()

        # attempt 1: blocked -> False,E1 已记入且容量预占被释放。
        self.assertFalse(scheduler._try_admit_prefill(runtime, 1_000))
        self.assertEqual(runtime.admission_evictions, (e1,))
        self.assertFalse(runtime.admitted_prefill)
        self.assertIn(
            ("release_request_capacity",
             ("session_0_request_0", 1_000), {}), kv_manager.calls)

        # capacity_epoch 变化(他处逐出触发 note)后重试:attempt 2 放行。
        scheduler._note_capacity_change(0)
        self.assertTrue(scheduler._try_admit_prefill(runtime, 2_000))

        # 核心断言:两次 attempt 的逐出累积,修复前此处为 (e2,)。
        self.assertEqual(runtime.admission_evictions, (e1, e2))
        self.assertTrue(runtime.admitted_prefill)


if __name__ == "__main__":
    unittest.main()
