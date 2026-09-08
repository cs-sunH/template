#!/usr/bin/env python3
"""test_sh20_task_load_snapshot.py -- 在线 task-load running 聚合偏差修复的
最小合成单元测试（背景见
《5仓库本该一致却不同排查报告.md》高-2 裁决）。

修复前：running 分量把在飞请求全部剩余聚合成一个"大 chunk"（context 取段末
全量）一次估算，attention ∝ chunk×context 被系统性抬高（在飞段越大越高），
与离线蓝本"逐 512-chunk 累加"及修复后的 sh_3.0 不符。修复后：running 与
queued 同为逐块求和、每请求恰计一次。

运行：cd sh_test_mesh/workload/llama2_7b_inference &&
      python3 online/test_sh20_task_load_snapshot.py   （或 pytest 同路径）
"""
import os
import sys
import types
import unittest
from dataclasses import make_dataclass
from unittest import mock

_ONLINE_DIR = os.path.dirname(os.path.abspath(__file__))
_WORKLOAD_DIR = os.path.dirname(_ONLINE_DIR)
for _p in (_ONLINE_DIR, _WORKLOAD_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from face_scheduler import (  # noqa: E402
    PREFILL_CHUNK_SIZE,
    FaceHardware,
    FaceInstanceSpec,
    FaceModel,
    FaceRequest,
    _validate_and_expand_requests,
    build_instances,
    estimate_decode_remaining_task_load_ns,
    estimate_prefill_task_load_ns,
)
from online.sh20_online_scheduler import (  # noqa: E402
    Sh20OnlineScheduler,
    _OnlineInstanceState,
)


def _make_hardware_model():
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
    return hardware, model


def _make_request(index: int, *, prefill_tokens: int = 1024,
                   decode_length: int = 8) -> FaceRequest:
    return FaceRequest(
        queue_index=index,
        session_id=f"s{index}",
        turn_index=0,
        request_id=f"r{index}",
        prefill_length=prefill_tokens,
        decode_length=decode_length,
        session_arrival_time_ns=0,
        inter_request_interval_ns=None,
    )


def _make_runtime(index: int, *, prefill_tokens: int = 1024):
    runtimes, _ = _validate_and_expand_requests(
        [_make_request(index, prefill_tokens=prefill_tokens)], PREFILL_CHUNK_SIZE)
    return runtimes[0]


class TaskLoadSnapshotExactlyOnceTest(unittest.TestCase):
    """恰计一次 / 分量归属 / 逐块口径（防聚合回归）三组断言。"""

    def setUp(self) -> None:
        hardware, model = _make_hardware_model()
        scheduler = Sh20OnlineScheduler.__new__(Sh20OnlineScheduler)
        scheduler.topology = build_instances(
            hardware,
            (
                FaceInstanceSpec("ins0", "1", (0, 1)),
                FaceInstanceSpec("ins1", "2", (2, 3)),
            ),
        )
        scheduler.instances = [
            _OnlineInstanceState(index=i)
            for i in range(len(scheduler.topology.instances))
        ]
        scheduler.p_chunk = PREFILL_CHUNK_SIZE
        scheduler.config = types.SimpleNamespace(hardware=hardware, model=model)
        scheduler._prefill_task_cache = {}
        scheduler._decode_task_load_cache = {}
        scheduler.average_decode_length = 10.0
        # 改法S2（快照纪元缓存 + qp 聚合账本，2026-08-23）：
        # _task_load_snapshot 的缓存包装读 _snapshot_verify，脚手架同步
        # 装配（缺省 False = 生产姿态）。
        scheduler._snapshot_verify = False
        self.scheduler = scheduler
        self.hardware = hardware
        self.model = model

    def _bind(self, *runtimes):
        """runtimes 挂到 scheduler.runtimes，返回各自 request_index（位次）。"""
        self.scheduler.runtimes = list(runtimes)
        return list(range(len(runtimes)))

    def _state(self, *, index, qp=(), busy=False, active_decode=(),
               prime=False):
        """拼 batch 改造（2026-08-22）重订：busy 门 = 一个列车在飞
        （in_flight_train 非 None）；running/queued 分量口径不变，仅
        状态字段随列车状态机改名。prime=True 时按改法S2-B 为 qp 成员
        装配聚合账本（生产路径由准入点 _try_admit_request 置全量）。"""
        state = _OnlineInstanceState(index=index)
        for request_index in qp:
            if prime:
                runtime = self.scheduler.runtimes[request_index]
                runtime.queued_chunk_load_ns = (
                    self.scheduler._queued_chunk_load_full_ns(
                        instance_size=(
                            self.scheduler.topology.instance(index).size),
                        runtime=runtime))
            state.qp.append(request_index)
        for request_index in active_decode:
            state.active_decode.append(request_index)
        state.in_flight_train = (
            {"train_id": "stub"} if busy else None)
        return state

    def test_inflight_counted_exactly_once(self):
        """同 token 形状（1024 = 2×512）请求：在飞态（busy、qp=[r0]）与纯
        排队态（不 busy、qp=[r1]）total_task_load_ns 必须相等——修复前
        running 聚合口径下两者不等（聚合 > 逐块），此断言失败。"""
        idx0, idx1 = self._bind(
            _make_runtime(0), _make_runtime(1))
        inflight = self._state(index=0, qp=(idx0,), busy=True, prime=True)
        queued = self._state(index=1, qp=(idx1,), busy=False, prime=True)
        snap_inflight = self.scheduler._task_load_snapshot(inflight, 0)
        snap_queued = self.scheduler._task_load_snapshot(queued, 0)
        self.assertGreater(snap_inflight.running_prefill_task_load_ns, 0)
        self.assertEqual(snap_inflight.queued_prefill_task_load_ns, 0)
        self.assertGreater(snap_queued.queued_prefill_task_load_ns, 0)
        self.assertEqual(snap_queued.running_prefill_task_load_ns, 0)
        self.assertEqual(
            snap_inflight.total_task_load_ns, snap_queued.total_task_load_ns)

    def test_components_split_inflight_vs_queued(self):
        """qp=[在飞 r0, 排队 r1]、busy=True：queued 分量 == 仅 r1 排队对照，
        running 分量 == 仅 r0 在飞对照（分量各自恰计一次）。"""
        idx0, idx1, idx1c, idx0c = self._bind(
            _make_runtime(0),
            _make_runtime(1, prefill_tokens=768),
            _make_runtime(2, prefill_tokens=768),
            _make_runtime(3),
        )
        composite = self._state(index=0, qp=(idx0, idx1), busy=True,
                                prime=True)
        queued_ctrl = self._state(index=1, qp=(idx1c,), busy=False,
                                  prime=True)
        running_ctrl = self._state(index=1, qp=(idx0c,), busy=True,
                                   prime=True)
        snap_composite = self.scheduler._task_load_snapshot(composite, 0)
        snap_queued = self.scheduler._task_load_snapshot(queued_ctrl, 0)
        snap_running = self.scheduler._task_load_snapshot(running_ctrl, 0)
        self.assertGreater(snap_queued.queued_prefill_task_load_ns, 0)
        self.assertGreater(snap_running.running_prefill_task_load_ns, 0)
        self.assertEqual(
            snap_composite.queued_prefill_task_load_ns,
            snap_queued.queued_prefill_task_load_ns)
        self.assertEqual(
            snap_composite.running_prefill_task_load_ns,
            snap_running.running_prefill_task_load_ns)

    def test_running_is_per_chunk_sum_not_aggregate(self):
        """防聚合回归：running 分量必须等于逐 512-chunk 手工求和；
        聚合口径（一个 2048 大 chunk、段末 context）严格大于该和。"""
        runtime = _make_runtime(0, prefill_tokens=2048)  # 4×512
        idx, = self._bind(runtime)
        state = self._state(index=0, qp=(idx,), busy=True, prime=True)
        snap = self.scheduler._task_load_snapshot(state, 0)
        manual = 0
        processed = 0
        remaining = runtime.prefill_tokens_to_process
        while remaining > 0:
            chunk = min(PREFILL_CHUNK_SIZE, remaining)
            manual += estimate_prefill_task_load_ns(
                hardware=self.hardware,
                model=self.model,
                instance_size=self.scheduler.topology.instance(0).size,
                chunk_tokens=chunk,
                context_tokens=runtime.history_tokens_before
                + processed + chunk,
            )
            processed += chunk
            remaining -= chunk
        self.assertEqual(snap.running_prefill_task_load_ns, manual)
        aggregate = estimate_prefill_task_load_ns(
            hardware=self.hardware,
            model=self.model,
            instance_size=self.scheduler.topology.instance(0).size,
            chunk_tokens=runtime.prefill_tokens_to_process,
            context_tokens=runtime.history_tokens_before
            + runtime.prefill_tokens_to_process,
        )
        self.assertGreater(aggregate, manual)

    def test_active_decode_uses_closed_form_remaining(self):
        """拼 batch 改造（2026-08-22）active decode 物理折算重订：
        旧口径"整段不可分（generated=0，剩余恒为全量 ADL）"作废——
        generated_tokens = decode_tokens_consumed（列车核销闭式推进），
        current_context_tokens = current_decode_token（= prefill_context +
        consumed）。断言：consumed 推进后 active 分量严格下降；且快照值 ==
        估算器按新口径的直接调用（公式/参数不动，仅折算输入换账本值）。"""
        from face_scheduler import estimate_decode_remaining_task_load_ns

        runtime = _make_runtime(0, prefill_tokens=512)
        runtime.decode_tokens_consumed = 0
        runtime.current_decode_token = runtime.prefill_context_tokens
        idx, = self._bind(runtime)
        state = self._state(index=0, active_decode=(idx,))
        snap_before = self.scheduler._task_load_snapshot(state, 0)

        # 列车核销闭式推进 5 个 token（决策边界口径）。改法S2-A：runtime
        # 进度字段属快照输入，生产由列车核销点 bump 实例纪元；本测试直接
        # 改账本，换新实例态取值（快照值只依赖账本字段，与实例态身份无关）。
        runtime.decode_tokens_consumed = 5
        runtime.current_decode_token = (
            runtime.prefill_context_tokens + 5)
        state_after = self._state(index=1, active_decode=(idx,))
        snap_after = self.scheduler._task_load_snapshot(state_after, 0)
        self.assertGreater(snap_before.active_decode_task_load_ns,
                           snap_after.active_decode_task_load_ns)
        expected = estimate_decode_remaining_task_load_ns(
            self.hardware, self.model,
            instance_size=self.scheduler.topology.instance(0).size,
            current_context_tokens=runtime.current_decode_token,
            generated_tokens=runtime.decode_tokens_consumed,
            average_decode_length=self.scheduler.average_decode_length,
            running_step_fraction_remaining=1.0)
        self.assertEqual(snap_after.active_decode_task_load_ns, expected)


class SnapshotEpochCacheTest(unittest.TestCase):
    """改法S2（2026-08-23，快照纪元缓存 + qp 聚合账本）簿记不变量与
    影子等价性（对齐 sh_3.0 母本 test_sh30_task_load_snapshot.py 的
    SnapshotEpochCacheTest 钉子姿态）。"""

    def setUp(self) -> None:
        hardware, model = _make_hardware_model()
        scheduler = Sh20OnlineScheduler.__new__(Sh20OnlineScheduler)
        scheduler.topology = build_instances(
            hardware,
            (
                FaceInstanceSpec("ins0", "1", (0, 1)),
                FaceInstanceSpec("ins1", "2", (2, 3)),
            ),
        )
        scheduler.instances = [
            _OnlineInstanceState(index=i)
            for i in range(len(scheduler.topology.instances))
        ]
        scheduler.p_chunk = PREFILL_CHUNK_SIZE
        scheduler.config = types.SimpleNamespace(hardware=hardware,
                                                 model=model)
        scheduler._prefill_task_cache = {}
        scheduler._decode_task_load_cache = {}
        scheduler.average_decode_length = 10.0
        scheduler._snapshot_verify = False
        self.scheduler = scheduler

    def test_cache_hit_reuses_and_bump_invalidates(self):
        """S2-A：纪元未变 → 命中同一快照对象；纪元 +1 → 重算（输入未变
        时数值恒等）。"""
        idx0, idx1 = self._bind(
            _make_runtime(0), _make_runtime(1))
        state = self._state(index=0, qp=(idx0,), busy=True, prime=True)
        snap1 = self.scheduler._task_load_snapshot(state, 0)
        snap2 = self.scheduler._task_load_snapshot(state, 0)  # 命中
        self.assertIs(snap2, snap1)
        self.scheduler._bump_instance_epoch(state)  # 模拟实例账本变更
        snap3 = self.scheduler._task_load_snapshot(state, 0)
        self.assertIsNot(snap3, snap1)
        self.assertEqual(snap3, snap1)  # 输入未变 → 数值恒等
        self.assertIsNot(idx1, None)  # bind 形状自检

    def _bind(self, *runtimes):
        """runtimes 挂到 scheduler.runtimes，返回各自 request_index（位次）。"""
        self.scheduler.runtimes = list(runtimes)
        return list(range(len(runtimes)))

    def _state(self, *, index, qp=(), busy=False, active_decode=(),
               prime=False):
        state = _OnlineInstanceState(index=index)
        for request_index in qp:
            if prime:
                runtime = self.scheduler.runtimes[request_index]
                runtime.queued_chunk_load_ns = (
                    self.scheduler._queued_chunk_load_full_ns(
                        instance_size=(
                            self.scheduler.topology.instance(index).size),
                        runtime=runtime))
            state.qp.append(request_index)
        for request_index in active_decode:
            state.active_decode.append(request_index)
        state.in_flight_train = (
            {"train_id": "stub"} if busy else None)
        return state

    def test_shadow_reference_equals_aggregate(self):
        """S2-B 影子：聚合路径 == 原始逐 chunk while 现算（组合态全覆盖：
        qp 多成员 + busy 在飞 + active_decode），命中路径同批再取零 raise。"""
        idx0, idx1, idxd = self._bind(
            _make_runtime(0, prefill_tokens=1536),
            _make_runtime(1, prefill_tokens=768),
            _make_runtime(2, prefill_tokens=512))
        d_runtime = self.scheduler.runtimes[idxd]
        d_runtime.decode_tokens_consumed = 0
        d_runtime.current_decode_token = d_runtime.prefill_context_tokens
        state = self._state(index=0, qp=(idx0, idx1), busy=True,
                            active_decode=(idxd,), prime=True)
        self.scheduler._snapshot_verify = True
        snapshot = self.scheduler._task_load_snapshot(state, 0)  # 零 raise
        reference = self.scheduler._reference_task_load_snapshot(state)
        self.assertEqual(snapshot, reference)
        again = self.scheduler._task_load_snapshot(state, 0)  # 命中路径
        self.assertEqual(again, reference)

    def test_ledger_constant_across_membership(self):
        """S2-B 恒定性：sh_2.0 口径下成员账本在 qp 会员期内恒定——
        模拟列车核销（remaining_chunks 推进 + busy 翻转）后账本与参照
        现算仍逐位一致。"""
        idx0, idx1 = self._bind(
            _make_runtime(0, prefill_tokens=1024),
            _make_runtime(1, prefill_tokens=512))
        state = self._state(index=0, qp=(idx0, idx1), busy=True,
                            prime=True)
        # 模拟列车核销：头部 remaining_chunks 推进 + busy 门翻转（生产
        # 由 _finalize_completed_trains bump 实例纪元，账本无扣减）。
        self.scheduler.runtimes[idx0].remaining_chunks = 0
        self.scheduler._bump_instance_epoch(state)
        reference = self.scheduler._reference_task_load_snapshot(state)
        compute = self.scheduler._compute_task_load_snapshot(state)
        self.assertEqual(compute, reference)
        # 头部 remaining_chunks=0 后不再是 running 候选：running 分量 =
        # 下一个有 chunk 工作的成员（idx1）账本，idx0 账本全额落入
        # queued（口径恰计一次不变）。
        self.assertEqual(
            compute.running_prefill_task_load_ns,
            self.scheduler.runtimes[idx1].queued_chunk_load_ns)
        self.assertEqual(
            compute.queued_prefill_task_load_ns,
            self.scheduler.runtimes[idx0].queued_chunk_load_ns)

class BoundedTaskLoadMemoTest(unittest.TestCase):
    """固定容量 memo：命中不重算，FIFO 淘汰只触发同函数精确重算。"""

    def setUp(self) -> None:
        hardware, model = _make_hardware_model()
        scheduler = Sh20OnlineScheduler.__new__(Sh20OnlineScheduler)
        scheduler.config = types.SimpleNamespace(hardware=hardware, model=model)
        scheduler._prefill_task_cache = {}
        scheduler._decode_task_load_cache = {}
        scheduler._task_load_cache_capacity = 2
        self.scheduler = scheduler
        self.hardware = hardware
        self.model = model

    def test_prefill_hit_and_fifo_eviction_preserve_exact_value(self):
        a = dict(instance_size=2, chunk_tokens=512, context_tokens=512)
        b = dict(instance_size=2, chunk_tokens=512, context_tokens=768)
        c = dict(instance_size=2, chunk_tokens=512, context_tokens=1024)
        expected = {
            tuple(args.values()): estimate_prefill_task_load_ns(
                hardware=self.hardware, model=self.model, **args)
            for args in (a, b, c)
        }
        with mock.patch(
                "online.sh20_online_scheduler.estimate_prefill_task_load_ns",
                wraps=estimate_prefill_task_load_ns) as estimate:
            self.assertEqual(self.scheduler._prefill_chunk_task_load_ns(**a),
                             expected[tuple(a.values())])
            self.assertEqual(self.scheduler._prefill_chunk_task_load_ns(**a),
                             expected[tuple(a.values())])  # hit
            self.assertEqual(self.scheduler._prefill_chunk_task_load_ns(**b),
                             expected[tuple(b.values())])
            self.assertEqual(self.scheduler._prefill_chunk_task_load_ns(**c),
                             expected[tuple(c.values())])  # evicts a
            self.assertEqual(self.scheduler._prefill_chunk_task_load_ns(**a),
                             expected[tuple(a.values())])  # exact recompute
        self.assertEqual(estimate.call_count, 4)
        self.assertEqual(len(self.scheduler._prefill_task_cache), 2)
        self.assertEqual(
            list(self.scheduler._prefill_task_cache),
            [tuple(c.values()), tuple(a.values())],
        )

    def test_decode_hit_and_fifo_eviction_preserve_exact_value(self):
        a = dict(instance_size=2, current_context_tokens=512,
                 generated_tokens=0, average_decode_length=10.0,
                 running_step_fraction_remaining=1.0)
        b = dict(instance_size=2, current_context_tokens=513,
                 generated_tokens=1, average_decode_length=10.0,
                 running_step_fraction_remaining=1.0)
        c = dict(instance_size=2, current_context_tokens=514,
                 generated_tokens=2, average_decode_length=10.0,
                 running_step_fraction_remaining=1.0)
        expected = {
            tuple(args.values()): estimate_decode_remaining_task_load_ns(
                self.hardware, self.model, **args)
            for args in (a, b, c)
        }
        with mock.patch(
                "online.sh20_online_scheduler.estimate_decode_remaining_task_load_ns",
                wraps=estimate_decode_remaining_task_load_ns) as estimate:
            self.assertEqual(self.scheduler._decode_task_load_ns_cached(**a),
                             expected[tuple(a.values())])
            self.assertEqual(self.scheduler._decode_task_load_ns_cached(**a),
                             expected[tuple(a.values())])  # hit
            self.assertEqual(self.scheduler._decode_task_load_ns_cached(**b),
                             expected[tuple(b.values())])
            self.assertEqual(self.scheduler._decode_task_load_ns_cached(**c),
                             expected[tuple(c.values())])  # evicts a
            self.assertEqual(self.scheduler._decode_task_load_ns_cached(**a),
                             expected[tuple(a.values())])  # exact recompute
        self.assertEqual(estimate.call_count, 4)
        self.assertEqual(len(self.scheduler._decode_task_load_cache), 2)
        self.assertEqual(
            list(self.scheduler._decode_task_load_cache),
            [tuple(c.values()), tuple(a.values())],
        )


if __name__ == "__main__":
    unittest.main()
