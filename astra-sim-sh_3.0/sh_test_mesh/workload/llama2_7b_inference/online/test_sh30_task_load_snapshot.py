#!/usr/bin/env python3
"""test_sh30_task_load_snapshot.py -- 在线 task_load 双计费修复的最小合成单元测试。

背景（《5仓库本该一致却不同排查报告.md》高-1，2026-08-20）：`Sh30OnlineScheduler.
_task_load_snapshot` 的 queued 循环遍历全部 `state.qp`（含在飞的 qp[0]），
且原 :897-900 的三元表达式两分支相同（死条件），prefill 在飞请求
（busy=True 且 qp[0].prefill_emitted=True）被 queued 与 running 各按
全量剩余计一次，≈ 2× 计入 total_task_load_ns（ordering_key 第一关键字），
实例选择系统性回避有在飞 prefill 的实例。修复：queued
循环跳过已发射请求——每请求恰计一次；在线 phase-1 近似 = 在飞全量
只进 running 分量（离线蓝本对应语义见修复后函数 docstring）。

运行：cd sh_test_mesh/workload/llama2_7b_inference &&
      python3 online/test_sh30_task_load_snapshot.py   （或 pytest 同路径）
"""
import os
import sys
import unittest

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
    build_instances,
)
from online.sh30_online_scheduler import (  # noqa: E402
    Sh30OnlineScheduler,
    _OnlineInstanceState,
    _OnlineRequestRuntime,
)


def _make_scheduler() -> Sh30OnlineScheduler:
    """绕过 __init__（需 manifest/graph/bridge），只设置 _task_load_snapshot
    用到的属性（小参数复用 test_face_scheduler.py 的 Roofline 测试配置）。"""
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
    scheduler.topology = build_instances(
        hardware,
        (
            FaceInstanceSpec("ins0", "1", (0, 1)),
            FaceInstanceSpec("ins1", "2", (2, 3)),
        ),
    )
    scheduler.instances = [
        _OnlineInstanceState(index=i) for i in range(len(scheduler.topology.instances))
    ]
    scheduler.p_chunk = PREFILL_CHUNK_SIZE
    scheduler.hardware = hardware
    scheduler.model = model
    scheduler.average_decode_length = 10.0
    scheduler._prefill_task_cache = {}
    return scheduler


def _make_runtime(request_id, *, history_tokens_before=0,
                  prefill_context_tokens=1024, decode_length=8):
    """record dict 提供 _OnlineRequestRuntime 必需字段；
    prefill_tokens_to_process = prefill_context_tokens - history_tokens_before。"""
    return _OnlineRequestRuntime({
        "request_id": request_id,
        "session_id": f"{request_id}_session",
        "turn_index": 0,
        "queue_index": 0,
        "prefill_length": prefill_context_tokens - history_tokens_before,
        "decode_length": decode_length,
        "history_tokens_before": history_tokens_before,
        "prefill_context_tokens": prefill_context_tokens,
        "final_context_tokens": prefill_context_tokens + decode_length,
    })


def _make_state(*, index, qp=(), busy=False, active_decode=()):
    state = _OnlineInstanceState(index=index)
    for runtime in qp:
        state.qp.append(runtime)
    for runtime in active_decode:
        state.active_decode.append(runtime)
        state.active_decode_lookup.add(runtime.request_id)
    state.busy = busy
    return state


class TaskLoadSnapshotExactlyOnceTest(unittest.TestCase):
    """方案 §2 场景 1-3：恰计一次 / queued 剔除在飞 / active_decode 不受影响。"""

    def setUp(self) -> None:
        self.scheduler = _make_scheduler()

    def test_inflight_request_counted_exactly_once(self):
        """场景 1（核心回归）：同一 token 形状（1024 = 2×512 chunk）的请求，
        在飞态（busy=True、qp=[r]、r.prefill_emitted=True）与纯排队态
        （busy=False、qp=[r']、未发射）的 total_task_load_ns 必须相等；
        修复前在飞态 ≈ 2×（queued 循环未剔除在飞），此断言失败。"""
        inflight = _make_runtime("r0")
        inflight.prefill_emitted = True
        state_inflight = _make_state(
            index=0, qp=(inflight,), busy=True)
        state_queued = _make_state(
            index=1, qp=(_make_runtime("r1"),), busy=False)
        snap_inflight = self.scheduler._task_load_snapshot(state_inflight, 0)
        snap_queued = self.scheduler._task_load_snapshot(state_queued, 0)
        # 在飞态：全量只进 running；纯排队态：全量只进 queued。
        self.assertGreater(snap_inflight.running_prefill_task_load_ns, 0)
        self.assertEqual(snap_inflight.queued_prefill_task_load_ns, 0)
        self.assertGreater(snap_queued.queued_prefill_task_load_ns, 0)
        self.assertEqual(snap_queued.running_prefill_task_load_ns, 0)
        self.assertEqual(
            snap_inflight.total_task_load_ns, snap_queued.total_task_load_ns)

    def test_queued_excludes_inflight_request(self):
        """场景 2：qp=[在飞 r0, 排队 r1]、busy=True 的组合态——
        queued 分量 == 仅 r1 排队的对照实例，running 分量 == 仅 r0 在飞的
        对照实例（分量各自恰计一次）。"""
        r0 = _make_runtime("r0", prefill_context_tokens=1024)
        r0.prefill_emitted = True
        r1 = _make_runtime("r1", prefill_context_tokens=768)
        composite = _make_state(index=0, qp=(r0, r1), busy=True)
        # 对照 1：仅 r1 排队（同 token 形状）。
        r1_only = _make_state(
            index=1, qp=(_make_runtime("r1c", prefill_context_tokens=768),),
            busy=False)
        # 对照 2：仅 r0 在飞（同 token 形状）。
        r0_ctrl = _make_runtime("r0c", prefill_context_tokens=1024)
        r0_ctrl.prefill_emitted = True
        r0_only = _make_state(index=1, qp=(r0_ctrl,), busy=True)
        snap_composite = self.scheduler._task_load_snapshot(composite, 0)
        snap_queued_ctrl = self.scheduler._task_load_snapshot(r1_only, 0)
        snap_running_ctrl = self.scheduler._task_load_snapshot(r0_only, 0)
        self.assertGreater(snap_queued_ctrl.queued_prefill_task_load_ns, 0)
        self.assertGreater(snap_running_ctrl.running_prefill_task_load_ns, 0)
        self.assertEqual(
            snap_composite.queued_prefill_task_load_ns,
            snap_queued_ctrl.queued_prefill_task_load_ns)
        self.assertEqual(
            snap_composite.running_prefill_task_load_ns,
            snap_running_ctrl.running_prefill_task_load_ns)

    def test_active_decode_unaffected_by_inflight_prefill(self):
        """场景 3：active_decode 分量只由 decode 成员决定——同一 active_decode
        集合在"实例有无在飞 prefill"两种状态下该分量逐位相等且 > 0。"""
        decode_qp_inflight = _make_runtime("r0")
        decode_qp_inflight.prefill_emitted = True
        with_prefill = _make_state(
            index=0, qp=(decode_qp_inflight,), busy=True,
            active_decode=(_make_runtime("d0"),))
        without_prefill = _make_state(
            index=1, active_decode=(_make_runtime("d1"),))
        snap_with = self.scheduler._task_load_snapshot(with_prefill, 0)
        snap_without = self.scheduler._task_load_snapshot(without_prefill, 0)
        self.assertGreater(snap_with.active_decode_task_load_ns, 0)
        self.assertGreater(snap_without.active_decode_task_load_ns, 0)
        self.assertEqual(
            snap_with.active_decode_task_load_ns,
            snap_without.active_decode_task_load_ns)


if __name__ == "__main__":
    unittest.main()
