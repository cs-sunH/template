#!/usr/bin/env python3
"""test_admit_gate.py -- 性能修复改法 A/D 的最小合成单元测试。

覆盖（SH系列性能修复执行文档 §2.1/§2.2，2026-08-22）：
  - 改法A（decode 估算全参 memo）：_decode_task_load_ns_cached 与直调
    estimate_decode_remaining_task_load_ns 数值恒等，且全参 key 命中缓存
    （重复调用不增长缓存字典，任一参数变化即分区新 key）；
  - 改法D（KV 账本纪元重试门）簿记不变量：纪元未变跳过重试（不调
    try_admit）、纪元 bump 后放行重试、失败记录失败纪元、成功即清除
    （有界性）、blocked FIFO 顺序不变、影子模式（SH_ADMIT_GATE_VERIFY=1）
    跳过分支仍完整评估并断言必返 False。

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

from face_scheduler import (  # noqa: E402
    FaceHardware,
    FaceModel,
    estimate_decode_remaining_task_load_ns,
)
from online.sh30_online_scheduler import (  # noqa: E402
    Sh30OnlineScheduler,
    _OnlineRequestRuntime,
)


def _make_scheduler() -> Sh30OnlineScheduler:
    """绕过 __init__（需 manifest/graph/bridge），只装配被测路径用到的
    属性（小参数 Roofline 配置与 test_sh30_task_load_snapshot.py 同款）。"""
    hardware = FaceHardware(
        mesh_rows=2,
        mesh_cols=2,
        local_hbm_capacity_bytes=1_000_000_000,
        local_hbm_bandwidth_gbps=100.0,
        d2d_bandwidth_gbps=200.0,
        peak_perf_tflops=1.0,
        d2d_latency_ns=0,
        local_hbm_latency_ns=0,
    )
    model = FaceModel(
        layers=2,
        hidden_size=16,
        ffn_size=32,
        num_heads=4,
        vocab_size=32,
        bytes_per_elem=2,
        mlp_variant="swiglu",
    )
    scheduler = Sh30OnlineScheduler.__new__(Sh30OnlineScheduler)
    scheduler.hardware = hardware
    scheduler.model = model
    scheduler._decode_task_load_cache = {}
    # 改法D 状态（__init__ 同款初值）。N6+F3+M2（2026-09-14 kimi 复审）：
    # 重试键改为 KV 纪元 ⊕ 失败候选集（选中 ∪ applicable 实例，冻结于
    # 失败时刻）纪元的 (kv, ((inst, epoch), ...)) 形态；夹具装配两实例。
    from online.sh30_online_scheduler import _OnlineInstanceState
    scheduler.instances = [_OnlineInstanceState(index=0),
                           _OnlineInstanceState(index=1)]
    scheduler._kv_ledger_epoch = 0
    scheduler._admit_attempt_epoch = {}
    scheduler._admission_failure_state = {}
    scheduler._admit_gate_verify = False
    scheduler.pending_admissions = deque()
    return scheduler


def _make_runtime(request_id):
    return _OnlineRequestRuntime({
        "request_id": request_id,
        "session_id": "%s_session" % request_id,
        "turn_index": 0,
        "queue_index": 0,
        "prefill_length": 1024,
        "decode_length": 8,
        "history_tokens_before": 0,
        "prefill_context_tokens": 1024,
        "final_context_tokens": 1032,
    })


class DecodeTaskLoadMemoTest(unittest.TestCase):
    """改法A：全参 key memo 与直调数学等价 + 缓存命中。"""

    def setUp(self) -> None:
        self.scheduler = _make_scheduler()

    def test_memo_equals_direct_call_and_hits_cache(self):
        """同参两次调用与直调数值恒等；缓存恰 1 条（第二次命中）。"""
        direct = estimate_decode_remaining_task_load_ns(
            self.scheduler.hardware, self.scheduler.model,
            instance_size=4,
            current_context_tokens=1024,
            generated_tokens=0,
            average_decode_length=10.0,
            running_step_fraction_remaining=1.0)
        first = self.scheduler._decode_task_load_ns_cached(
            instance_size=4, current_context_tokens=1024,
            generated_tokens=0, average_decode_length=10.0,
            running_step_fraction_remaining=1.0)
        second = self.scheduler._decode_task_load_ns_cached(
            instance_size=4, current_context_tokens=1024,
            generated_tokens=0, average_decode_length=10.0,
            running_step_fraction_remaining=1.0)
        self.assertEqual(first, direct)
        self.assertEqual(second, direct)
        self.assertEqual(len(self.scheduler._decode_task_load_cache), 1)

    def test_full_param_key_partitions(self):
        """key 含全部估算入参：任一参数变化即新 key（阶段 3 进度分区
        预留），两 key 共存于缓存。"""
        for kwargs in (
            dict(instance_size=4, current_context_tokens=1024,
                 generated_tokens=0, average_decode_length=10.0,
                 running_step_fraction_remaining=1.0),
            dict(instance_size=4, current_context_tokens=768,
                 generated_tokens=0, average_decode_length=10.0,
                 running_step_fraction_remaining=1.0),
            dict(instance_size=4, current_context_tokens=1024,
                 generated_tokens=3, average_decode_length=10.0,
                 running_step_fraction_remaining=1.0),
        ):
            self.scheduler._decode_task_load_ns_cached(**kwargs)
        self.assertEqual(len(self.scheduler._decode_task_load_cache), 3)


class AdmitGateLedgerEpochTest(unittest.TestCase):
    """改法D：纪元重试门簿记不变量（_try_admit_request 打桩隔离）。"""

    def setUp(self) -> None:
        self.scheduler = _make_scheduler()

    def _stub(self, results):
        """results: request_id -> bool（缺省 False）；记录调用序列。失败
        路径同步留 _last_admit_failure_key（生产语义：失败候选集纪元键，
        两实例候选集）。"""
        calls = []

        def fake_try_admit(runtime, now_ns):
            calls.append(runtime.request_id)
            if results.get(runtime.request_id, False):
                return True
            self.scheduler._last_admit_failure_key = (
                self.scheduler._kv_ledger_epoch,
                ((0, self.scheduler.instances[0].ledger_epoch),
                 (1, self.scheduler.instances[1].ledger_epoch)))
            return False

        self.scheduler._try_admit_request = fake_try_admit
        return calls

    def test_bump_increments_epoch(self):
        self.assertEqual(self.scheduler._kv_ledger_epoch, 0)
        self.scheduler._bump_kv_ledger_epoch()
        self.assertEqual(self.scheduler._kv_ledger_epoch, 1)

    def test_skip_when_epoch_unchanged(self):
        """上次失败键 == 当前键（KV 纪元与候选集纪元均未变）→ 跳过
        重试（打桩被调即失败）。"""
        r0 = _make_runtime("r0")
        self.scheduler.pending_admissions.append(r0)
        self.scheduler._kv_ledger_epoch = 5
        self.scheduler._admit_attempt_epoch["r0"] = (5, ((0, 0), (1, 0)))
        calls = self._stub({})
        self.scheduler._admit_waiting_requests(0)
        self.assertEqual(calls, [])
        self.assertEqual(list(self.scheduler.pending_admissions), [r0])

    def test_instance_epoch_alone_reopens_gate(self):
        """N6 核心场景：KV 纪元未变但失败**选中**实例的负载纪元迁移 →
        重开重试门（argmin 换选时机；旧纯 KV 纪元门会错跳过）。"""
        r0 = _make_runtime("r0")
        self.scheduler.pending_admissions.append(r0)
        self.scheduler._kv_ledger_epoch = 5
        self.scheduler._admit_attempt_epoch["r0"] = (5, ((0, 0), (1, 0)))
        self.scheduler.instances[0].ledger_epoch = 1  # 选中实例负载迁移
        calls = self._stub({})
        self.scheduler._admit_waiting_requests(0)
        self.assertEqual(calls, ["r0"])
        self.assertEqual(
            self.scheduler._admit_attempt_epoch["r0"], (5, ((0, 1), (1, 0))))
        # 键稳定后再次 pass：跳过（无自旋）。
        self.scheduler._admit_waiting_requests(0)
        self.assertEqual(calls, ["r0"])

    def test_candidate_instance_epoch_alone_reopens_gate(self):
        """M2 核心场景（kimi 复审）：KV 纪元与选中实例纪元均未变，但
        失败候选集内**未选中**实例负载迁移（argmin 可能翻转到该可行
        候选）→ 必须重开重试门；旧"仅选中实例"键会错跳过（时机损失 +
        SH_ADMIT_GATE_VERIFY 影子断言被合法击穿）。"""
        r0 = _make_runtime("r0")
        self.scheduler.pending_admissions.append(r0)
        self.scheduler._kv_ledger_epoch = 5
        self.scheduler._admit_attempt_epoch["r0"] = (5, ((0, 0), (1, 0)))
        self.scheduler.instances[1].ledger_epoch = 2  # 候选实例负载迁移
        calls = self._stub({})
        self.scheduler._admit_waiting_requests(0)
        self.assertEqual(calls, ["r0"])
        self.assertEqual(
            self.scheduler._admit_attempt_epoch["r0"], (5, ((0, 0), (1, 2))))
        # 候选集外实例（不在键内）迁移不重开：键不含实例 2 语义由
        # _current_retry_key 的冻结集保证（此处两实例全在集内，稳定
        # 后再跑一次验证无自旋）。
        self.scheduler._admit_waiting_requests(0)
        self.assertEqual(calls, ["r0"])

    def test_retry_after_epoch_bump_and_record(self):
        """KV 纪元变化 → 放行重试；失败记录新键，随后同键再跳过。"""
        r0 = _make_runtime("r0")
        self.scheduler.pending_admissions.append(r0)
        self.scheduler._kv_ledger_epoch = 1
        self.scheduler._admit_attempt_epoch["r0"] = (0, ((0, 0), (1, 0)))
        calls = self._stub({})
        self.scheduler._admit_waiting_requests(0)
        self.assertEqual(calls, ["r0"])
        self.assertEqual(
            self.scheduler._admit_attempt_epoch["r0"], (1, ((0, 0), (1, 0))))
        self.assertEqual(list(self.scheduler.pending_admissions), [r0])
        # 同键第二个 pass：跳过（无新调用）。
        self.scheduler._admit_waiting_requests(0)
        self.assertEqual(calls, ["r0"])

    def test_success_clears_attempt_epoch(self):
        """成功准入即清除条目（有界性：dict 不随时间增长）。"""
        r0 = _make_runtime("r0")
        self.scheduler.pending_admissions.append(r0)
        self.scheduler._kv_ledger_epoch = 1
        self.scheduler._admit_attempt_epoch["r0"] = (0, ((0, 0), (1, 0)))
        self._stub({"r0": True})
        self.scheduler._admit_waiting_requests(0)
        self.assertEqual(list(self.scheduler.pending_admissions), [])
        self.assertNotIn("r0", self.scheduler._admit_attempt_epoch)
        self.assertEqual(self.scheduler._admit_attempt_epoch, {})

    def test_blocked_fifo_order_preserved(self):
        """混合键的 blocked 批重排后 FIFO 顺序不变。"""
        r0, r1, r2 = (_make_runtime(i) for i in ("r0", "r1", "r2"))
        for r in (r0, r1, r2):
            self.scheduler.pending_admissions.append(r)
        self.scheduler._kv_ledger_epoch = 2
        self.scheduler._admit_attempt_epoch["r0"] = (2, ((0, 0), (1, 0)))  # 跳过
        self.scheduler._admit_attempt_epoch["r1"] = (0, ((0, 0), (1, 0)))  # 重试（失败）
        # r2 无记录 → 重试（失败）
        calls = self._stub({})
        self.scheduler._admit_waiting_requests(0)
        self.assertEqual(calls, ["r1", "r2"])
        self.assertEqual(
            list(self.scheduler.pending_admissions), [r0, r1, r2])

    def test_shadow_mode_full_evaluation_and_assertion(self):
        """影子模式：跳过分支仍完整评估；返回 False 零 violation，
        返回 True 必 raise（等价性断言 fail-closed）。"""
        r0 = _make_runtime("r0")
        self.scheduler.pending_admissions.append(r0)
        self.scheduler._kv_ledger_epoch = 3
        self.scheduler._admit_attempt_epoch["r0"] = (3, ((0, 0), (1, 0)))
        self.scheduler._admit_gate_verify = True
        calls = self._stub({"r0": False})
        self.scheduler._admit_waiting_requests(0)
        self.assertEqual(calls, ["r0"])  # 完整评估（零收益、全检查）
        self.assertEqual(
            self.scheduler._admit_attempt_epoch["r0"], (3, ((0, 0), (1, 0))))
        self.assertEqual(list(self.scheduler.pending_admissions), [r0])
        # 违例形态：门判跳过的重试居然成功 → RuntimeError。
        self.scheduler.pending_admissions = deque([r0])
        self._stub({"r0": True})
        with self.assertRaises(RuntimeError) as ctx:
            self.scheduler._admit_waiting_requests(0)
        self.assertIn("admit gate equivalence violated", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
