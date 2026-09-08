#!/usr/bin/env python3
"""test_ablation_switch.py -- TaskA 消融开关（SH30_ABLATION，作者裁定
2026-08-26）的 python 级单测（不跑仿真）。

覆盖（每档 >= 2 个构造性用例 + fail-closed）：
  - 开关解析：4 个合法档位；非法值 _parse_ablation_mode raise；
    Sh30OnlineScheduler.__init__ 非法 env fail-closed（基类打桩）；
    各档 __init__ 派生旗标（no_lb/no_affinity 布尔）。
  - none 档（缺省等价）：_select_prefill_instance 与直调
    select_prefill_instance 逐值恒等；三分支（1a/1/3）选择与现树一致；
    分支2 sticky；驻留不可行仍返回 False；drain decode == prefill 实例
    （decode_hbm_feasible_instances 零调用）。
  - no_lb 档：_select_prefill_instance 退化为掩码内首选可行（min 索引，
    负载不进判据）；空掩码 fail-closed；分支 1a/1/3 调用点退化断言。
  - no_affinity 档：分支2 LOCAL sticky 退化为全集均衡选择；PARTIAL 钉扎
    边界不动；drain decode 走均衡路径（real KVCacheManager：跨实例
    noc_migrate + joiner 入 decode 实例 pending_decode_ready + 预约迁移/
    释放 + 会话搬家）；驻留不可行但全集可行时不再等待。
  - no_lb_no_affinity 档：两旗标同置；分支2/decode 选择随 no_lb 取首选
    可行（decode_hbm_feasible_instances 仍被调用——消融路径证明）。

运行：cd sh_test_mesh/workload/llama2_7b_inference &&
      python3 online/test_ablation_switch.py   （或 pytest 同路径）
"""
import os
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

_ONLINE_DIR = os.path.dirname(os.path.abspath(__file__))
_WORKLOAD_DIR = os.path.dirname(_ONLINE_DIR)
for _p in (_ONLINE_DIR, _WORKLOAD_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import online.sh30_online_scheduler as sh30_module  # noqa: E402
from face_scheduler import (  # noqa: E402
    PREFILL_CHUNK_SIZE,
    FaceHardware,
    FaceInstanceSpec,
    FaceModel,
    InstanceTaskLoadSnapshot,
    KVCacheManager,
    build_instances,
    kv_cache_shard_bytes_for_layer_range,
    select_prefill_instance,
)
from online.online_scheduler_base import OnlineSchedulerBase  # noqa: E402
from online.sh30_online_scheduler import (  # noqa: E402
    Sh30OnlineScheduler,
    _ABLATION_MODES,
    _OnlineInstanceState,
    _OnlineRequestRuntime,
    _first_feasible_instance,
    _parse_ablation_mode,
)


# ---------------------------------------------------------------------------
# 夹具（Roofline 小参数与 test_sh30_task_load_snapshot.py / test_face_
# scheduler.py 同款；调度器一律 __new__ 壳式装配，只装被测路径属性）。
# ---------------------------------------------------------------------------

def _tiny_hardware(capacity_bytes=1_000_000_000):
    return FaceHardware(
        mesh_rows=2,
        mesh_cols=2,
        local_hbm_capacity_bytes=capacity_bytes,
        local_hbm_bandwidth_gbps=100.0,
        d2d_bandwidth_gbps=200.0,
        peak_perf_tflops=1.0,
        d2d_latency_ns=0,
        local_hbm_latency_ns=0,
    )


def _tiny_model(layers=4):
    return FaceModel(
        layers=layers,
        hidden_size=16,
        ffn_size=32,
        num_heads=4,
        vocab_size=32,
        bytes_per_elem=2,
        mlp_variant="swiglu",
    )


def _two_instance_topology(hardware):
    return build_instances(
        hardware,
        (
            FaceInstanceSpec("ins0", "1", (0, 1)),
            FaceInstanceSpec("ins1", "2", (2, 3)),
        ),
    )


class _GraphStub:
    def sync_pending_history_after_evictions(self, evictions):
        pass


class _AdmitKVStub:
    """_try_admit_request 的 KV 打桩：可行性掩码/历史快照全部受控。"""

    def __init__(self, feasible, history=None, session_ids=()):
        self._feasible = tuple(feasible)
        self._history = history
        self._session_ids = set(session_ids)
        self.has_session_calls = 0

    @property
    def session_ids(self):
        raise AssertionError(
            "admission membership must not materialize sorted session_ids")

    def has_session(self, session_id):
        self.has_session_calls += 1
        return session_id in self._session_ids

    def request_hbm_feasible_instances(self, **kwargs):
        return self._feasible

    def request_hbm_eventually_feasible_instances(self, **kwargs):
        return tuple(True for _ in self._feasible)

    def session_snapshot(self, session_id):
        return self._history

    def reserve_request_capacity(self, **kwargs):
        return ()

    def prepare_prefill(self, **kwargs):
        return (None, None, ())


def _shell_scheduler(kv_manager, *, no_lb=False, no_affinity=False,
                     topology=None, hardware=None, model=None):
    """装配 _try_admit_request/_on_prefill_drain 用到的全部属性。"""
    hardware = hardware or _tiny_hardware()
    model = model or _tiny_model()
    topology = topology or _two_instance_topology(hardware)
    scheduler = Sh30OnlineScheduler.__new__(Sh30OnlineScheduler)
    scheduler.topology = topology
    scheduler.hardware = hardware
    scheduler.model = model
    scheduler.p_chunk = PREFILL_CHUNK_SIZE
    scheduler.average_decode_length = 10.0
    scheduler._prefill_task_cache = {}
    scheduler._decode_task_load_cache = {}
    scheduler._snapshot_verify = False
    scheduler.kv_manager = kv_manager
    scheduler.edge_free_mask = (False, False)
    scheduler.edge_mask = (True, True)
    scheduler.instances = [
        _OnlineInstanceState(index=i)
        for i in range(len(topology.instances))
    ]
    scheduler.runtime_by_request_id = {}
    scheduler._kv_ledger_epoch = 0
    scheduler._ready_frontier = set()
    scheduler._ablation_no_lb = no_lb
    scheduler._ablation_no_affinity = no_affinity
    scheduler.graph = _GraphStub()
    scheduler._ledger_admit = lambda *args, **kwargs: None
    scheduler._emit_admission = lambda *args, **kwargs: None
    scheduler._emit_join_decision = lambda *args, **kwargs: None
    return scheduler


def _make_runtime(request_id, *, prefill_context_tokens=1024,
                  decode_length=8, history_tokens_before=0):
    return _OnlineRequestRuntime(
        {
            "request_id": request_id,
            "session_id": "%s_session" % request_id,
            "turn_index": 0,
            "queue_index": 0,
            "prefill_length": (
                prefill_context_tokens - history_tokens_before),
            "decode_length": decode_length,
            "history_tokens_before": history_tokens_before,
            "prefill_context_tokens": prefill_context_tokens,
            "final_context_tokens": (
                prefill_context_tokens + decode_length),
        },
        PREFILL_CHUNK_SIZE,
    )


def _add_queued_load(scheduler, instance_index, request_id):
    """在指定实例的 qp 里制造排队负载（快照三分量 queued > 0）。"""
    runtime = _make_runtime(request_id)
    scheduler.runtime_by_request_id[request_id] = runtime
    scheduler.instances[instance_index].qp.append(runtime)
    runtime.queued_chunk_load_ns = scheduler._queued_chunk_load_full_ns(
        instance_size=(
            scheduler.topology.instance(instance_index).size),
        runtime=runtime,
    )
    return runtime


def _admit_once(scheduler, request_id):
    runtime = _make_runtime(request_id)
    scheduler.runtime_by_request_id[request_id] = runtime
    runtime.estimated_arrival_ns = 0
    admitted = scheduler._try_admit_request(runtime, 0)
    return admitted, runtime


# ---------------------------------------------------------------------------
# 开关解析与 fail-closed
# ---------------------------------------------------------------------------

class AblationModeParseTest(unittest.TestCase):
    """_parse_ablation_mode：合法档位/非法值 fail-closed。"""

    def test_all_four_modes_parse(self):
        for mode in _ABLATION_MODES:
            self.assertEqual(_parse_ablation_mode(mode), mode)

    def test_invalid_values_fail_closed(self):
        for bad in ("", "full", "None", "NO_LB", "no_lb ",
                    "no_lb,no_affinity", "no_lb+no_affinity", "1"):
            with self.assertRaises(ValueError) as ctx:
                _parse_ablation_mode(bad)
            self.assertIn("SH30_ABLATION", str(ctx.exception))


class SchedulerInitAblationTest(unittest.TestCase):
    """Sh30OnlineScheduler.__init__：一次读取 + 非法值 fail-closed +
    档位→旗标派生（基类/拓扑/KV 账本打桩——开关解析位于 p_chunk 校验后、
    拓扑/KV 构建之前）。"""

    @staticmethod
    def _construct(env_value):
        """env_value=None 表示删除变量（缺省档）。"""
        config = SimpleNamespace(
            prefill_chunk_size=PREFILL_CHUNK_SIZE,
            inference_groups=(),
            kv_reserve_context_tokens=0,
            model=None,
            hardware=None,
            source_average_decode_length=10.0,
        )
        with mock.patch.object(
                OnlineSchedulerBase, "__init__",
                lambda self, **kwargs: None), \
             mock.patch.object(sh30_module, "build_instances"), \
             mock.patch.object(sh30_module, "KVCacheManager"), \
             mock.patch.object(sh30_module, "edge_free_instance_mask",
                               return_value=(False,)), \
             mock.patch.object(sh30_module, "edge_instance_mask",
                               return_value=(False,)), \
             mock.patch.dict(os.environ):
            if env_value is None:
                os.environ.pop("SH30_ABLATION", None)
            else:
                os.environ["SH30_ABLATION"] = env_value
            return Sh30OnlineScheduler(
                manifest={"requests": []},
                config=config,
                graph=SimpleNamespace(
                    set_next_plan=lambda mapping: None,
                    set_plan_resolver=lambda resolver: None,
                ),
                mode="strategy",
            )

    def test_default_and_none_mode_flags(self):
        for env_value in (None, "none"):
            scheduler = self._construct(env_value)
            self.assertEqual(scheduler._ablation, "none")
            self.assertFalse(scheduler._ablation_no_lb)
            self.assertFalse(scheduler._ablation_no_affinity)

    def test_mode_flag_derivation(self):
        expectations = {
            "no_lb": (True, False),
            "no_affinity": (False, True),
            "no_lb_no_affinity": (True, True),
        }
        for mode, (no_lb, no_affinity) in expectations.items():
            scheduler = self._construct(mode)
            self.assertEqual(scheduler._ablation, mode)
            self.assertEqual(scheduler._ablation_no_lb, no_lb)
            self.assertEqual(scheduler._ablation_no_affinity, no_affinity)

    def test_invalid_env_fails_closed_at_init(self):
        for bad in ("full", "", "no_lb_affinity"):
            with self.assertRaises(ValueError) as ctx:
                self._construct(bad)
            self.assertIn("SH30_ABLATION", str(ctx.exception))


# ---------------------------------------------------------------------------
# no_lb：select_prefill_instance 调用点退化
# ---------------------------------------------------------------------------

def _snapshot(index, total_load_ns):
    return InstanceTaskLoadSnapshot(
        instance_index=index,
        running_prefill_task_load_ns=total_load_ns,
        queued_prefill_task_load_ns=0,
        active_decode_task_load_ns=0,
        last_arrival_ns=None,
    )


class SelectPrefillInstanceGateTest(unittest.TestCase):
    """_select_prefill_instance 消融门：none 恒等直调 / no_lb 首选可行。"""

    def setUp(self):
        self.kv = _AdmitKVStub((True, True))
        # 负载不对称：实例 0 重、实例 1 轻（ordering_key 轻者胜）。
        self.snapshots = (_snapshot(0, 5000), _snapshot(1, 100))

    def test_none_mode_equals_direct_call(self):
        """none 档路径等价：包装函数与 face_scheduler 原函数逐值恒等
        （多组掩码/负载下构造性对拍）。"""
        scheduler = _shell_scheduler(self.kv, no_lb=False)
        for mask in ((True, True), (False, True), (True, False)):
            self.assertEqual(
                scheduler._select_prefill_instance(
                    self.snapshots, mask),
                select_prefill_instance(self.snapshots, mask))

    def test_no_lb_picks_first_feasible_ignoring_load(self):
        scheduler = _shell_scheduler(self.kv, no_lb=True)
        self.assertEqual(
            scheduler._select_prefill_instance(
                self.snapshots, (True, True)), 0)
        self.assertEqual(
            scheduler._select_prefill_instance(
                self.snapshots, (False, True)), 1)
        # 负载反转不改变结果（min 索引判据，负载不进判据）。
        reversed_snapshots = (_snapshot(0, 0), _snapshot(1, 9000))
        self.assertEqual(
            scheduler._select_prefill_instance(
                reversed_snapshots, (True, True)), 0)

    def test_first_feasible_helper_fail_closed_on_empty_mask(self):
        with self.assertRaises(ValueError):
            _first_feasible_instance((False, False))
        scheduler = _shell_scheduler(self.kv, no_lb=True)
        with self.assertRaises(ValueError):
            scheduler._select_prefill_instance(
                self.snapshots, (False, False))


class TryAdmitNoLbTest(unittest.TestCase):
    """_try_admit_request 三个 select 调用点的 no_lb 退化断言
    （none 档对照 = 均衡选轻载实例 1；no_lb 档 = 首选可行实例 0）。"""

    def _run(self, *, no_lb, edge_free_mask, history):
        kv = _AdmitKVStub(
            (True, True), history=history,
            session_ids=() if history is None else ("new_session",))
        scheduler = _shell_scheduler(kv, no_lb=no_lb)
        scheduler.edge_free_mask = edge_free_mask
        _add_queued_load(scheduler, 0, "heavy0")
        return _admit_once(scheduler, "new")

    def test_membership_uses_one_direct_lookup(self):
        """会话存在性判定不得为一次准入排序整个活跃会话集合。"""
        remote = SimpleNamespace(location="remote_memory")
        kv = _AdmitKVStub(
            (True, True), history=remote, session_ids=("new_session",))
        scheduler = _shell_scheduler(kv)

        admitted, _ = _admit_once(scheduler, "new")

        self.assertTrue(admitted)
        self.assertEqual(kv.has_session_calls, 1)

    def test_branch_1a_full_set_lb_degenerates(self):
        """分支 1a（无非边缘实例→全集均衡）：none 选轻载 1，no_lb 选 0。"""
        admitted, runtime = self._run(
            no_lb=False, edge_free_mask=(False, False), history=None)
        self.assertTrue(admitted)
        self.assertEqual(runtime.prefill_instance_index, 1)
        self.assertEqual(
            runtime.prefill_affinity_reason, "first_request_edge_fallback")
        admitted, runtime = self._run(
            no_lb=True, edge_free_mask=(False, False), history=None)
        self.assertTrue(admitted)
        self.assertEqual(runtime.prefill_instance_index, 0)

    def test_branch_1_non_edge_lb_degenerates(self):
        """分支 1（首请求非边缘，掩码 = 可行∧非边缘）。"""
        admitted, runtime = self._run(
            no_lb=False, edge_free_mask=(True, True), history=None)
        self.assertTrue(admitted)
        self.assertEqual(runtime.prefill_instance_index, 1)
        self.assertEqual(
            runtime.prefill_affinity_reason, "first_request_non_edge")
        admitted, runtime = self._run(
            no_lb=True, edge_free_mask=(True, True), history=None)
        self.assertTrue(admitted)
        self.assertEqual(runtime.prefill_instance_index, 0)

    def test_branch_3_remote_edge_lb_degenerates(self):
        """分支 3（REMOTE 历史→边缘集合均衡）。"""
        remote = SimpleNamespace(location="remote_memory")
        admitted, runtime = self._run(
            no_lb=False, edge_free_mask=(False, False), history=remote)
        self.assertTrue(admitted)
        self.assertEqual(runtime.prefill_instance_index, 1)
        self.assertEqual(
            runtime.prefill_affinity_reason, "remote_edge_load_balance")
        admitted, runtime = self._run(
            no_lb=True, edge_free_mask=(False, False), history=remote)
        self.assertTrue(admitted)
        self.assertEqual(runtime.prefill_instance_index, 0)


# ---------------------------------------------------------------------------
# no_affinity：分支2 prefill 退化 + PARTIAL 边界 + drain decode 均衡
# ---------------------------------------------------------------------------

class TryAdmitNoAffinityTest(unittest.TestCase):

    def test_branch_2_local_sticky_degenerates_to_balanced(self):
        """分支2 LOCAL 驻留：none 钉驻留 0；no_affinity 全集均衡选轻载 1。"""
        local = SimpleNamespace(location="local_hbm", instance_index=0)
        kv = _AdmitKVStub((True, True), history=local,
                          session_ids=("new_session",))
        scheduler = _shell_scheduler(kv, no_affinity=False)
        _add_queued_load(scheduler, 0, "heavy0")
        admitted, runtime = _admit_once(scheduler, "new")
        self.assertTrue(admitted)
        self.assertEqual(runtime.prefill_instance_index, 0)
        self.assertEqual(
            runtime.prefill_affinity_reason, "resident_local_hbm")

        scheduler = _shell_scheduler(kv, no_affinity=True)
        _add_queued_load(scheduler, 0, "heavy0")
        admitted, runtime = _admit_once(scheduler, "new")
        self.assertTrue(admitted)
        self.assertEqual(runtime.prefill_instance_index, 1)
        # 词表冻结：reason 仍按驻留状态记。
        self.assertEqual(
            runtime.prefill_affinity_reason, "resident_local_hbm")

    def test_branch_2_partial_pinning_boundary_untouched(self):
        """PARTIAL 钉扎是 KV 不变量（声明边界）：no_affinity 档仍钉驻留
        实例（全集可行也不搬家）。"""
        partial = SimpleNamespace(
            location="partial_hbm_remote", instance_index=1)
        kv = _AdmitKVStub((True, True), history=partial,
                          session_ids=("new_session",))
        scheduler = _shell_scheduler(kv, no_affinity=True)
        # 实例 1 重载：均衡若被采用会选 0——钉扎必须胜出。
        _add_queued_load(scheduler, 1, "heavy1")
        admitted, runtime = _admit_once(scheduler, "new")
        self.assertTrue(admitted)
        self.assertEqual(runtime.prefill_instance_index, 1)
        self.assertEqual(
            runtime.prefill_affinity_reason, "resident_prefix_layers")

    def test_resident_infeasible_wait_only_in_none_mode(self):
        """驻留实例不可行：none 档维持等待（False）；no_affinity 档退化为
        全集选择 → 改选可行实例 1（语义差异即消融点）。"""
        local = SimpleNamespace(location="local_hbm", instance_index=0)
        kv = _AdmitKVStub((False, True), history=local,
                          session_ids=("new_session",))
        scheduler = _shell_scheduler(kv, no_affinity=False)
        admitted, runtime = _admit_once(scheduler, "new")
        self.assertFalse(admitted)
        self.assertIsNone(runtime.prefill_instance_index)

        scheduler = _shell_scheduler(kv, no_affinity=True)
        admitted, runtime = _admit_once(scheduler, "new")
        self.assertTrue(admitted)
        self.assertEqual(runtime.prefill_instance_index, 1)

    def test_branch_2_no_lb_no_affinity_picks_first_feasible(self):
        """no_lb_no_affinity：分支2 退化为全集首选可行（驻留 1、可行集
        (True,True) → 0，证明 sticky 已关）。"""
        local = SimpleNamespace(location="local_hbm", instance_index=1)
        kv = _AdmitKVStub((True, True), history=local,
                          session_ids=("new_session",))
        scheduler = _shell_scheduler(
            kv, no_lb=True, no_affinity=True)
        admitted, runtime = _admit_once(scheduler, "new")
        self.assertTrue(admitted)
        self.assertEqual(runtime.prefill_instance_index, 0)


class DrainDecodeAffinityTest(unittest.TestCase):
    """_on_prefill_drain 的 decode 选择（real KVCacheManager，全 KV 调用
    走真实路径）。夹具：r0 已在实例 0 准入（预约+会话+qp），实例 0 另有
    排队负载（重），实例 1 空。"""

    def _make_scheduler(self, *, no_lb=False, no_affinity=False):
        hardware = _tiny_hardware()
        model = _tiny_model()
        topology = _two_instance_topology(hardware)
        kv = KVCacheManager(topology, model)
        scheduler = _shell_scheduler(
            kv, no_lb=no_lb, no_affinity=no_affinity,
            topology=topology, hardware=hardware, model=model)
        return scheduler, kv

    def _admit_on_instance0(self, scheduler, kv):
        """手工补齐 _try_admit_request 在实例 0 准入 r0 的全部账本。"""
        runtime = _make_runtime("r0")
        scheduler.runtime_by_request_id["r0"] = runtime
        kv.reserve_request_capacity(
            request_id="r0",
            session_id=runtime.session_id,
            instance_index=0,
            final_context_tokens=runtime.final_context_tokens,
        )
        kv.prepare_prefill(
            session_id=runtime.session_id,
            target_instance_index=0,
            history_tokens=0,
            trigger_request_id="r0",
        )
        kv.expand_prefill(
            session_id=runtime.session_id,
            instance_index=0,
            context_tokens=runtime.prefill_context_tokens,
            trigger_request_id="r0",
        )
        runtime.prefill_instance_index = 0
        scheduler.instances[0].qp.append(runtime)
        runtime.queued_chunk_load_ns = 0  # drain 的 fail-closed 断言要求 0
        return runtime

    def _spy_decode_feasibility(self, kv):
        calls = []
        original = kv.decode_hbm_feasible_instances

        def spy(**kwargs):
            calls.append(kwargs)
            return original(**kwargs)

        kv.decode_hbm_feasible_instances = spy
        return calls

    def test_none_mode_decode_stays_on_prefill_instance(self):
        """none 档：decode 固定同实例（红线 #4）；可行性查询零调用
        （纯门卫——消融分支不执行）；迁移 local_hit。"""
        scheduler, kv = self._make_scheduler(no_affinity=False)
        runtime = self._admit_on_instance0(scheduler, kv)
        _add_queued_load(scheduler, 0, "heavy0")
        calls = self._spy_decode_feasibility(kv)
        scheduler._on_prefill_drain("r0", 1000)
        self.assertEqual(calls, [])
        self.assertEqual(runtime.decode_instance_index, 0)
        self.assertEqual(
            [r.request_id
             for r in scheduler.instances[0].pending_decode_ready],
            ["r0"])
        self.assertEqual(
            scheduler.instances[1].pending_decode_ready, [])
        self.assertEqual(
            runtime.prefill_decode_transfer.kind, "local_hit")
        self.assertEqual(
            kv.session_snapshot(runtime.session_id).instance_index, 0)

    def test_no_lb_alone_keeps_decode_sticky(self):
        """no_lb 档只退化 prefill 调用点：decode 仍同实例（可行性零调用）。"""
        scheduler, kv = self._make_scheduler(no_lb=True)
        runtime = self._admit_on_instance0(scheduler, kv)
        _add_queued_load(scheduler, 0, "heavy0")
        calls = self._spy_decode_feasibility(kv)
        scheduler._on_prefill_drain("r0", 1000)
        self.assertEqual(calls, [])
        self.assertEqual(runtime.decode_instance_index, 0)

    def test_no_affinity_decode_balanced_to_light_instance(self):
        """no_affinity 档：decode 走均衡路径选轻载实例 1——跨实例 KV 全链
        （预约迁移 + noc_migrate 整上下文 + expand_decode(target) + 预约
        释放），joiner 入 decode 实例 pending_decode_ready，会话搬家。"""
        scheduler, kv = self._make_scheduler(no_affinity=True)
        runtime = self._admit_on_instance0(scheduler, kv)
        _add_queued_load(scheduler, 0, "heavy0")
        self.assertGreater(
            scheduler._task_load_snapshot(
                scheduler.instances[0], 1000).total_task_load_ns,
            scheduler._task_load_snapshot(
                scheduler.instances[1], 1000).total_task_load_ns)
        calls = self._spy_decode_feasibility(kv)
        scheduler._on_prefill_drain("r0", 1000)
        # 均衡路径被走（可行性查询恰一次，携带本请求预约排除）。
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["session_id"], runtime.session_id)
        self.assertEqual(
            calls[0]["final_context_tokens"],
            runtime.final_context_tokens)
        self.assertEqual(calls[0]["reservation_request_id"], "r0")
        self.assertEqual(runtime.decode_instance_index, 1)
        self.assertEqual(
            [r.request_id
             for r in scheduler.instances[1].pending_decode_ready],
            ["r0"])
        self.assertEqual(
            scheduler.instances[0].pending_decode_ready, [])
        transfer = runtime.prefill_decode_transfer
        self.assertEqual(transfer.kind, "noc_migrate")
        self.assertEqual(
            transfer.reason, "prefill_decode_instance_migrate")
        self.assertEqual(transfer.source_instance_index, 0)
        self.assertEqual(transfer.target_instance_index, 1)
        # 整上下文 KV shard 手算对拍（本夹具 = prefill 上下文全层）。
        expected = sum(kv_cache_shard_bytes_for_layer_range(
            scheduler.model, runtime.prefill_context_tokens,
            kv.tp_degree, layer_start=0,
            layer_end=scheduler.model.layers))
        self.assertEqual(transfer.total_bytes, expected)
        self.assertEqual(
            kv.session_snapshot(runtime.session_id).instance_index, 1)
        self.assertNotIn("r0", kv._reservations)  # 预约已核销
        self.assertEqual(0, scheduler.instances[0].qp.count(runtime))

    def test_no_lb_no_affinity_decode_picks_first_feasible(self):
        """no_lb_no_affinity 档：decode 选择随 no_lb 取首选可行（min 索引
        0，与均衡选择 1 相区分）；可行性查询仍被调用（消融路径证明）。"""
        scheduler, kv = self._make_scheduler(
            no_lb=True, no_affinity=True)
        runtime = self._admit_on_instance0(scheduler, kv)
        _add_queued_load(scheduler, 0, "heavy0")
        calls = self._spy_decode_feasibility(kv)
        scheduler._on_prefill_drain("r0", 1000)
        self.assertEqual(len(calls), 1)
        self.assertEqual(runtime.decode_instance_index, 0)
        self.assertEqual(
            [r.request_id
             for r in scheduler.instances[0].pending_decode_ready],
            ["r0"])
        self.assertEqual(
            runtime.prefill_decode_transfer.kind, "local_hit")


if __name__ == "__main__":
    unittest.main()
