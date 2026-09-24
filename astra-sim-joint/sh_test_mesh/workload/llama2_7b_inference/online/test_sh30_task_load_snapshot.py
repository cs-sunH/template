#!/usr/bin/env python3
"""test_sh30_task_load_snapshot.py -- task_load 快照"恰计一次"钉子
（拼 batch 改造重订，2026-08-22）。

背景（双计费修复史 + 列车化重订归因）：原修复（2026-08-20，高-1）钉住
"在飞 prefill 请求恰计一次"——queued 循环剔除已发射请求（prefill_emitted
门）。拼 batch 列车化重构后该口径按新账本重订：prefill 主体移入实例迭代
列车，发射门 = in_flight_train；在飞 chunk 负载 = 冻结列车账本
（prefill_chunk_spans，running 分量），其余剩余 chunk 自
prefill_tokens_completed 闭式折算（queued 分量，恰计一次——running 自
queued 扣除）；active_decode 分量的"active 段不可分（generated=0/
fraction=1.0 全量剩余）"假设作废，改为 decode_tokens_consumed 闭式迭代级
剩余量（打分公式与阈值不动，红线；task-load 只改物理折算）。

场景：
  1. 在飞列车覆盖队列头全部剩余 chunk ⇒ total 与纯排队态逐位相等
     （恰计一次的核心回归）；
  2. 组合态（在飞 r0 + 排队 r1）queued/running 分量各自与对照实例一致；
  3. active_decode 分量不受实例有无在飞列车影响，且随 decode_tokens_
     consumed 闭式推进单调下降（迭代级剩余量）。

运行：cd sh_test_mesh/workload/llama2_7b_inference &&
      python3 online/test_sh30_task_load_snapshot.py   （或 pytest 同路径）
"""
import os
import sys
import unittest
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
    build_instances,
    estimate_decode_remaining_task_load_ns,
    estimate_prefill_task_load_ns,
)
from online.sh30_online_scheduler import (  # noqa: E402
    Sh30OnlineScheduler,
    _OnlineInstanceState,
    _OnlineRequestRuntime,
    _TASK_LOAD_CACHE_CAPACITY,  # F6 销账：替身对齐 __init__ 初值
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
    # N12（R15-4）甄别改写：average_decode_length 标定常数 → 在线因果
    # 估计器（无观测样本时冷启动缺省 10 token，数值与旧常数路径一致，
    # 既有期望值全部保持）。
    from joint.joint_cost_model import CausalHorizonEstimator
    scheduler._joint_horizon = CausalHorizonEstimator(
        cold_start_default_tokens=10)
    scheduler._prefill_task_cache = {}
    # 改法A（decode 估算全参 memo）：_task_load_snapshot 的 active_decode 段
    # 改读 _decode_task_load_ns_cached，脚手架同步装配其缓存字典。
    scheduler._decode_task_load_cache = {}
    # 对齐 __init__ 初值（F6 销账：软门已删，替身漏设 = AttributeError；
    # BoundedTaskLoadMemoTest.setUp 覆写为 2 验证 FIFO 语义）。
    scheduler._task_load_cache_capacity = _TASK_LOAD_CACHE_CAPACITY
    # 改法S2（快照纪元缓存 + qp 聚合账本，2026-08-23）：_task_load_snapshot
    # 的缓存包装读 _snapshot_verify，脚手架同步装配（缺省 False = 生产姿态）。
    scheduler._snapshot_verify = False
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
    }, PREFILL_CHUNK_SIZE)


def _train_with_chunk_spans(spans):
    """在飞列车桩（_task_load_snapshot 只读 prefill_chunk_spans）。"""
    return {
        "train_id": "stub",
        "iterations": len(spans),
        "members": [],
        "exit_set": set(),
        "drain_set": set(),
        "prefill_chunk_tokens": [],
        "prefill_chunk_spans": list(spans),
        "pass_spans": [],
    }


def _make_state(*, index, qp=(), in_flight_train=None, active_decode=(),
                scheduler=None):
    state = _OnlineInstanceState(index=index)
    for runtime in qp:
        # 改法S2-B（qp 聚合账本）：手工态同步装配成员聚合账本（生产路径
        # 由准入点 _try_admit_request 置全量，此处等价补齐）。
        if scheduler is not None:
            runtime.queued_chunk_load_ns = (
                scheduler._queued_chunk_load_full_ns(
                    instance_size=(
                        scheduler.topology.instance(index).size),
                    runtime=runtime))
        state.qp.append(runtime)
    for runtime in active_decode:
        state.active_decode.append(runtime)
        state.active_decode_lookup.add(runtime)
    state.in_flight_train = in_flight_train
    return state


class TaskLoadSnapshotExactlyOnceTest(unittest.TestCase):
    """场景 1-3：恰计一次 / 分量分离 / active_decode 迭代级闭式折算。"""

    def setUp(self) -> None:
        self.scheduler = _make_scheduler()

    def test_inflight_request_counted_exactly_once(self):
        """场景 1（核心回归）：同一 token 形状（1024 = 2×512 chunk）的请求，
        在飞态（列车覆盖其全部剩余 chunk）与纯排队态（无列车）的
        total_task_load_ns 必须相等；在飞态全量只进 running 分量。"""
        inflight = _make_runtime("r0")
        state_inflight = _make_state(
            index=0, qp=(inflight,), scheduler=self.scheduler,
            in_flight_train=_train_with_chunk_spans(
                [(512, 512), (512, 1024)]))
        state_queued = _make_state(
            index=1, qp=(_make_runtime("r1"),), scheduler=self.scheduler)
        snap_inflight = self.scheduler._task_load_snapshot(state_inflight, 0)
        snap_queued = self.scheduler._task_load_snapshot(state_queued, 0)
        # 在飞态：全量只进 running；纯排队态：全量只进 queued。
        self.assertGreater(snap_inflight.running_prefill_task_load_ns, 0)
        self.assertEqual(snap_inflight.queued_prefill_task_load_ns, 0)
        self.assertGreater(snap_queued.queued_prefill_task_load_ns, 0)
        self.assertEqual(snap_queued.running_prefill_task_load_ns, 0)
        self.assertEqual(
            snap_inflight.total_task_load_ns, snap_queued.total_task_load_ns)

    def test_queued_excludes_inflight_train_chunks(self):
        """场景 2：qp=[在飞 r0, 排队 r1]、列车覆盖 r0 的组合态——
        queued 分量 == 仅 r1 排队的对照实例，running 分量 == 仅 r0 在飞的
        对照实例（分量各自恰计一次）。"""
        r0 = _make_runtime("r0", prefill_context_tokens=1024)
        r1 = _make_runtime("r1", prefill_context_tokens=768)
        composite = _make_state(
            index=0, qp=(r0, r1), scheduler=self.scheduler,
            in_flight_train=_train_with_chunk_spans(
                [(512, 512), (512, 1024)]))
        # 对照 1：仅 r1 排队（同 token 形状）。
        r1_only = _make_state(
            index=1, qp=(_make_runtime("r1c", prefill_context_tokens=768),),
            scheduler=self.scheduler)
        # 对照 2：仅 r0 在飞（同 token 形状）。
        r0_ctrl = _make_runtime("r0c", prefill_context_tokens=1024)
        r0_only = _make_state(
            index=1, qp=(r0_ctrl,), scheduler=self.scheduler,
            in_flight_train=_train_with_chunk_spans(
                [(512, 512), (512, 1024)]))
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

    def test_active_decode_unaffected_by_inflight_train(self):
        """场景 3a：active_decode 分量只由 decode 成员决定——同一
        active_decode 集合在"实例有无在飞列车"两种状态下该分量逐位相等
        且 > 0。"""
        with_train = _make_state(
            index=0, qp=(_make_runtime("r0"),), scheduler=self.scheduler,
            in_flight_train=_train_with_chunk_spans([(512, 512)]),
            active_decode=(_make_runtime("d0"),))
        without_train = _make_state(
            index=1, active_decode=(_make_runtime("d1"),))
        snap_with = self.scheduler._task_load_snapshot(with_train, 0)
        snap_without = self.scheduler._task_load_snapshot(without_train, 0)
        self.assertGreater(snap_with.active_decode_task_load_ns, 0)
        self.assertGreater(snap_without.active_decode_task_load_ns, 0)
        self.assertEqual(
            snap_with.active_decode_task_load_ns,
            snap_without.active_decode_task_load_ns)

    def test_active_decode_iteration_level_closed_form(self):
        """场景 3b（新口径归因）：active_decode 剩余量随 decode_tokens_
        consumed 闭式推进单调下降（迭代级剩余量，替代原 generated=0
        全量剩余的不可分假设）；推进量 = ADL 上限内的剩余迭代。"""
        fresh = _make_runtime("d0", decode_length=8)
        advanced = _make_runtime("d1", decode_length=8)
        advanced.decode_tokens_consumed = 5
        advanced.current_decode_token = (
            advanced.prefill_context_tokens + 5)
        snap_fresh = self.scheduler._task_load_snapshot(
            _make_state(index=0, active_decode=(fresh,)), 0)
        snap_advanced = self.scheduler._task_load_snapshot(
            _make_state(index=1, active_decode=(advanced,)), 0)
        self.assertGreater(
            snap_fresh.active_decode_task_load_ns,
            snap_advanced.active_decode_task_load_ns,
            "closed-form remaining load must shrink as "
            "decode_tokens_consumed advances")
        # ADL=10：fresh 剩 10 个迭代估价、advanced 剩 5 个——两者的
        # active 分量都为正（advanced 仍有剩余，不归零）。
        self.assertGreater(snap_advanced.active_decode_task_load_ns, 0)
        exhausted = _make_runtime("d2", decode_length=8)
        exhausted.decode_tokens_consumed = 12  # 超过 ADL：估价剩余为 0
        snap_exhausted = self.scheduler._task_load_snapshot(
            _make_state(index=0, active_decode=(exhausted,)), 0)
        self.assertEqual(snap_exhausted.active_decode_task_load_ns, 0)


class SnapshotEpochCacheTest(unittest.TestCase):
    """改法S2（2026-08-23，快照纪元缓存 + qp 聚合账本）簿记不变量与
    影子等价性（对齐 test_admit_gate.py 对改法D 的钉子姿态）。"""

    def setUp(self) -> None:
        self.scheduler = _make_scheduler()

    def test_cache_hit_reuses_and_bump_invalidates(self):
        """S2-A：纪元未变 → 命中同一快照对象；纪元 +1 → 重算（输入未变
        时数值恒等）。"""
        r0 = _make_runtime("r0")
        state = _make_state(index=0, qp=(r0,), scheduler=self.scheduler)
        snap1 = self.scheduler._task_load_snapshot(state, 0)
        snap2 = self.scheduler._task_load_snapshot(state, 0)  # 命中
        self.assertIs(snap2, snap1)
        self.scheduler._bump_instance_epoch(state)  # 模拟实例账本变更
        snap3 = self.scheduler._task_load_snapshot(state, 0)
        self.assertIsNot(snap3, snap1)
        self.assertEqual(snap3, snap1)  # 输入未变 → 数值恒等

    def test_shadow_reference_equals_aggregate(self):
        """S2-B 影子：聚合路径 == 原始逐 chunk while 现算（组合态全覆盖：
        qp 多成员 + 在飞列车 + active_decode），命中路径同批再取零 raise。"""
        r0 = _make_runtime("r0", prefill_context_tokens=1536)
        r1 = _make_runtime("r1", prefill_context_tokens=768)
        state = _make_state(
            index=0, qp=(r0, r1), scheduler=self.scheduler,
            in_flight_train=_train_with_chunk_spans([(512, 512)]),
            active_decode=(_make_runtime("d0"),))
        self.scheduler._snapshot_verify = True
        snapshot = self.scheduler._task_load_snapshot(state, 0)  # 零 raise
        reference = self.scheduler._reference_task_load_snapshot(state)
        self.assertEqual(snapshot, reference)
        again = self.scheduler._task_load_snapshot(state, 0)  # 命中路径
        self.assertEqual(again, reference)

    def test_train_finalize_decrement_is_exact(self):
        """S2-B 扣减精确性：列车核销推进一个 chunk 后，成员聚合账本余量
        == 原始 while 现算的剩余贡献（整数恒等的逐点钉子）。"""
        r0 = _make_runtime("r0", prefill_context_tokens=1536)
        state = _make_state(index=0, qp=(r0,), scheduler=self.scheduler)
        # 模拟 _finalize_completed_trains 的核销推进（首 chunk 512 token：
        # span 键 = (512, history+0+512)）。
        r0.prefill_tokens_completed += 512
        r0.remaining_chunks -= 1
        r0.queued_chunk_load_ns -= self.scheduler._prefill_chunk_task_load_ns(
            instance_size=self.scheduler.topology.instance(0).size,
            chunk_tokens=512, context_tokens=512)
        reference = self.scheduler._reference_task_load_snapshot(state)
        self.assertEqual(
            reference.queued_prefill_task_load_ns, r0.queued_chunk_load_ns)

class BoundedTaskLoadMemoTest(unittest.TestCase):
    """固定容量 memo：命中不重算，FIFO 淘汰只触发同函数精确重算。"""

    def setUp(self) -> None:
        self.scheduler = _make_scheduler()
        self.scheduler._task_load_cache_capacity = 2

    def test_prefill_hit_and_fifo_eviction_preserve_exact_value(self):
        a = dict(instance_size=2, chunk_tokens=512, context_tokens=512)
        b = dict(instance_size=2, chunk_tokens=512, context_tokens=768)
        c = dict(instance_size=2, chunk_tokens=512, context_tokens=1024)
        expected = {
            tuple(args.values()): estimate_prefill_task_load_ns(
                self.scheduler.hardware, self.scheduler.model, **args)
            for args in (a, b, c)
        }
        with mock.patch(
                "online.sh30_online_scheduler.estimate_prefill_task_load_ns",
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
                self.scheduler.hardware, self.scheduler.model, **args)
            for args in (a, b, c)
        }
        with mock.patch(
                "online.sh30_online_scheduler.estimate_decode_remaining_task_load_ns",
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
