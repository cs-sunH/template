#!/usr/bin/env python3
"""test_admit_gate.py -- 改法D（KV 账本纪元重试门）的最小合成单元测试。

被测语义（sh20_online_scheduler._admit_pass 头部循环，key=request_index）：
  1. 上次失败纪元 == 当前 KV 账本纪元 → 跳过重试（不调 _try_admit_request，
     FIFO 位置不变）；纪元 bump 后恢复重试；
  2. 重试成功 → _admit_attempt_epoch 清除该条目且不回 pending（有界性）；
  3. 影子模式（_admit_gate_verify=True）→ 被门跳过的条目仍完整重评估；
     重试返回 True 即等价性破坏，fail-closed raise；
  4. _bump_kv_ledger_epoch 恰好 +1。

构造方式与 test_sh20_task_load_snapshot.py 同款（__new__ + 手工绑定，
不完整构造调度器；_try_admit_request 用可探单桩替换，只测门逻辑本身）。

运行：cd sh_test_mesh/workload/llama2_7b_inference &&
      python3 online/test_admit_gate.py   （或 pytest 同路径）
"""
import os
import sys
import unittest
from collections import deque

_ONLINE_DIR = os.path.dirname(os.path.abspath(__file__))
_WORKLOAD_DIR = os.path.dirname(_ONLINE_DIR)
for _p in (_ONLINE_DIR, _WORKLOAD_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from online.sh20_online_scheduler import (  # noqa: E402
    Sh20OnlineScheduler,
)


class _AdmitRecorder:
    """_try_admit_request 单桩：按 request_index 查表返回，记录调用序。"""

    def __init__(self, outcomes):
        self.outcomes = dict(outcomes)
        self.calls = []

    def __call__(self, request_index, now_ns):
        self.calls.append(request_index)
        return self.outcomes[request_index]


class AdmitGateEpochRetryTest(unittest.TestCase):

    def setUp(self) -> None:
        scheduler = Sh20OnlineScheduler.__new__(Sh20OnlineScheduler)
        scheduler.pending_admissions = deque()
        scheduler._kv_ledger_epoch = 0
        scheduler._admit_attempt_epoch = {}
        scheduler._admit_gate_verify = False
        scheduler.instances = []  # _admit_pass 尾部发射循环遍历空集
        self.scheduler = scheduler

    def test_gate_skips_retry_until_epoch_bumps(self):
        """同纪元二次 _admit_pass 不重试；bump 后恢复重试且 FIFO 序不变。"""
        rec = _AdmitRecorder({0: False, 1: False})
        self.scheduler._try_admit_request = rec
        self.scheduler.pending_admissions.extend([0, 1])
        self.scheduler._admit_pass(100)
        self.assertEqual(rec.calls, [0, 1])
        self.assertEqual(list(self.scheduler.pending_admissions), [0, 1])
        self.assertEqual(self.scheduler._admit_attempt_epoch, {0: 0, 1: 0})

        # 纪元未变：门跳过（零重试），FIFO 位置不变。
        self.scheduler._admit_pass(200)
        self.assertEqual(rec.calls, [0, 1])
        self.assertEqual(list(self.scheduler.pending_admissions), [0, 1])

        # 纪元 bump（任一 KV 变更点之后）：两条目均恢复重试。
        self.scheduler._bump_kv_ledger_epoch()
        self.scheduler._admit_pass(300)
        self.assertEqual(rec.calls, [0, 1, 0, 1])
        self.assertEqual(self.scheduler._admit_attempt_epoch, {0: 1, 1: 1})

    def test_success_clears_entry_and_leaves_pending(self):
        """成功准入：条目从 _admit_attempt_epoch 清除且不回 pending。"""
        rec = _AdmitRecorder({0: False, 1: True, 2: False})
        self.scheduler._try_admit_request = rec
        self.scheduler.pending_admissions.extend([0, 1, 2])
        self.scheduler._admit_pass(100)
        self.assertEqual(rec.calls, [0, 1, 2])
        self.assertEqual(list(self.scheduler.pending_admissions), [0, 2])
        self.assertEqual(self.scheduler._admit_attempt_epoch, {0: 0, 2: 0})
        self.assertNotIn(1, self.scheduler._admit_attempt_epoch)

    def test_shadow_mode_retries_skipped_entries(self):
        """影子模式：门判候选仍完整重评估（零收益、全检查）。"""
        rec = _AdmitRecorder({0: False})
        self.scheduler._try_admit_request = rec
        self.scheduler._admit_gate_verify = True
        self.scheduler.pending_admissions.extend([0])
        self.scheduler._admit_pass(100)
        self.scheduler._admit_pass(200)  # 同纪元也必须重评估
        self.assertEqual(rec.calls, [0, 0])
        self.assertEqual(self.scheduler._admit_attempt_epoch, {0: 0})
        self.assertEqual(list(self.scheduler.pending_admissions), [0])

    def test_shadow_mode_violation_raises(self):
        """影子断言：跳过条目重试返回 True → 等价性破坏 fail-closed。"""
        rec = _AdmitRecorder({0: False})
        self.scheduler._try_admit_request = rec
        self.scheduler._admit_gate_verify = True
        self.scheduler.pending_admissions.extend([0])
        self.scheduler._admit_pass(100)
        rec.outcomes[0] = True  # 模拟门判错误（账本未变却可准入）
        with self.assertRaisesRegex(
                RuntimeError, "admit gate equivalence violated"):
            self.scheduler._admit_pass(200)

    def test_bump_helper_increments_by_one(self):
        """_bump_kv_ledger_epoch 恰好 +1（9 个变更点各调一次的原子步长）。"""
        self.assertEqual(self.scheduler._kv_ledger_epoch, 0)
        self.scheduler._bump_kv_ledger_epoch()
        self.assertEqual(self.scheduler._kv_ledger_epoch, 1)
        self.scheduler._bump_kv_ledger_epoch()
        self.assertEqual(self.scheduler._kv_ledger_epoch, 2)


if __name__ == "__main__":
    unittest.main()
