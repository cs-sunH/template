#!/usr/bin/env python3
"""test_sh20_task_load_snapshot.py -- 在线 task-load running 聚合偏差修复的
最小合成单元测试（《sh_2.0偏差修改的执行方案.md》；背景见
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
        scheduler.average_decode_length = 10.0
        self.scheduler = scheduler
        self.hardware = hardware
        self.model = model

    def _bind(self, *runtimes):
        """runtimes 挂到 scheduler.runtimes，返回各自 request_index（位次）。"""
        self.scheduler.runtimes = list(runtimes)
        return list(range(len(runtimes)))

    def _state(self, *, index, qp=(), busy=False, active_decode=()):
        state = _OnlineInstanceState(index=index)
        for request_index in qp:
            state.qp.append(request_index)
        for request_index in active_decode:
            state.active_decode.append(request_index)
        state.busy = busy
        return state

    def test_inflight_counted_exactly_once(self):
        """同 token 形状（1024 = 2×512）请求：在飞态（busy、qp=[r0]）与纯
        排队态（不 busy、qp=[r1]）total_task_load_ns 必须相等——修复前
        running 聚合口径下两者不等（聚合 > 逐块），此断言失败。"""
        idx0, idx1 = self._bind(
            _make_runtime(0), _make_runtime(1))
        inflight = self._state(index=0, qp=(idx0,), busy=True)
        queued = self._state(index=1, qp=(idx1,), busy=False)
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
        composite = self._state(index=0, qp=(idx0, idx1), busy=True)
        queued_ctrl = self._state(index=1, qp=(idx1c,), busy=False)
        running_ctrl = self._state(index=1, qp=(idx0c,), busy=True)
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
        state = self._state(index=0, qp=(idx,), busy=True)
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


if __name__ == "__main__":
    unittest.main()
