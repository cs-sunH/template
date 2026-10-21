#!/usr/bin/env python3
"""test_joint_tie_rotation_wiring.py -- v9（2026-10-20，平局裁决轮转
语义）SH 侧（sh30_online_scheduler）接线单测。

joint 主臂跨实例平局 = 按实例编号环序轮转分摊（joint_scheduler 的
``joint_tie_rotation_pick`` 纯函数，子任务 A 已冻结契约）。本文件钉
SH 侧四组接线：

1. **配额重选镜像**：可行集缩集时 ``_quota_reselect_feasible`` 与
   主臂同构（游标非 0 时选环序在集成员；全集含被配额剔除实例时
   沿环序跳过不在集者）；cost_model=None 替身缺省臂同构；顺序模式
   分支不轮转；单实例动作平局不轮转（argmin 同旧序）。
2. **推进判据**（``_consume_joint_tie_rotation``）：joint 模式多实例
   平局推进到下一编号（含越最大值回绕）；非 joint 模式
   （load-first/face_static）不推进；forced 命中不推进；唯一最优/
   选择不在并列面不推进。
3. **全路径**（真实 ``_try_admit_request``，选择器打桩——native 投影
   面与本接线正交）：成功行三键取决策起点捕获游标（推进发生在有效
   选择之后，成员现值已前移）；容量失败（KVCapacityError）照推且
   失败首行披露捕获游标值、同请求第二次失败折叠 wait 行无三键；
   quota-deferred 提前 return 在强制钩子/推进点之前（游标不动、
   失败首行无三键）。
4. **L2 键含游标**：``_joint_arrival_selection`` 缓存键并入
   ``_joint_tie_cursor`` 分量——推进后同 request_view 重选必 miss
   重算（缓存统计 + 选择器调用计数双验）。

选择器打桩实现真实契约（argmin + joint_tie_rotation_pick + 诊断三
键）——打桩的是计价遍历（native 面），选择语义钉的是被测接线本身。

运行：cd sh_test_mesh/workload/llama2_7b_inference &&
      python3 online/test_joint_tie_rotation_wiring.py   （或 pytest 同路径）
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

from face_scheduler import (  # noqa: E402
    FaceHardware,
    FaceInstanceSpec,
    FaceModel,
    KVCacheManager,
    build_instances,
)
from joint.joint_config import (  # noqa: E402
    SCHEDULER_MODE_FACE_STATIC,
    SCHEDULER_MODE_LOAD_FIRST,
)
from joint.joint_cost_model import (  # noqa: E402
    ActionCandidate,
    CausalHorizonEstimator,
    InstanceLoadView,
    JointHardwareRates,
    LinkFlowRegistry,
    RequestView,
    ServiceFactors,
    SessionKVView,
)
from joint.joint_scheduler import (  # noqa: E402
    SelectionRecord,
    joint_tie_rotation_pick,
)
from joint.link_quota import (  # noqa: E402
    LinkQuotaTracker,
    QuotaVerdict,
)
from online.sh30_online_scheduler import (  # noqa: E402
    _OnlineInstanceState,
    _OnlineRequestRuntime,
    _PoolPortRegistry,
    _TASK_LOAD_CACHE_CAPACITY,
    Sh30OnlineScheduler,
)
import online.sh30_online_scheduler as _sh30_module  # noqa: E402


# ============================================================ 夹具构造 ==

def _candidate(instance_index, action, *, applicable=True, cost_ns=1000):
    """真实 ActionCandidate（dataclass——_quota_filter_candidates 经
    dataclasses.replace 改写不适用候选，替身必须同型）。"""
    return ActionCandidate(
        instance_index=instance_index, action=action,
        applicable=applicable,
        inapplicable_reason=(None if applicable else "not applicable"),
        cost_ns=(cost_ns if applicable else None),
        breakdown=None,
    )


def _tie_record(instances=(0, 1, 2), *, stay_cost=500, copy_cost=900,
                chosen_instance=None, mode="joint"):
    """跨实例 stay 平局的标准候选表（stay 全并列最优、copy 恒更贵）。

    chosen_instance=None 时按 (cost, order_key) argmin（= 最小编号
    stay）；显式给定用于构造"强制顶到并列面外/非 argmin 选择"形态。"""
    candidates = []
    for index in instances:
        candidates.append(_candidate(index, "stay", cost_ns=stay_cost))
        candidates.append(_candidate(index, "copy", cost_ns=copy_cost))
    applicable = [item for item in candidates if item.applicable]
    chosen = min(applicable, key=lambda item: (
        item.cost_ns, item.order_key()))
    if chosen_instance is not None:
        chosen = next(
            item for item in applicable
            if item.instance_index == chosen_instance)
    return SelectionRecord(
        mode=mode, chosen=chosen, candidates=tuple(candidates),
        instance_rule_note="joint instance x action argmin")


def _bare_scheduler(*, mode="joint", cursor=0):
    """__new__ 范式（C8/C14 同款）：只装配被测路径用到的属性。"""
    scheduler = Sh30OnlineScheduler.__new__(Sh30OnlineScheduler)
    scheduler._joint_mode = mode
    scheduler._joint_tie_cursor = cursor
    return scheduler


def _load_views(*instances):
    views = {}
    for index in instances:
        views[index] = InstanceLoadView(
            instance_index=index,
            queued_task_load_ns=0, running_task_load_ns=0,
            active_decode_task_load_ns=0,
            hbm_remaining_bytes_by_tp_rank=(1, 1),
            reclaimable_bytes_by_tp_rank=(0, 0))
    return views


class _GraphStub:
    """_try_admit_request 成功路径所需构图器最小替身（仅记录调用）。"""

    def __init__(self) -> None:
        self.synced = []
        self.admission_plans = []

    def sync_pending_history_after_evictions(self, transfers):
        self.synced.append(transfers)

    def emit_admission_batch(self, plan):
        self.admission_plans.append(plan)
        return {"eviction_watches": []}


def _full_scheduler():
    """绕过 __init__，装配 _try_admit_request 全路径用到的属性
    （test_joint_decision_schema._make_scheduler 同款骨架 + v9 轮转
    游标与强制钩子属性）。"""
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
        (FaceInstanceSpec("ins0", "1", (0, 1)),
         FaceInstanceSpec("ins1", "2", (2, 3))),
    )
    scheduler.instances = [
        _OnlineInstanceState(index=i)
        for i in range(len(scheduler.topology.instances))]
    scheduler.hardware = hardware
    scheduler.model = model
    scheduler.kv_manager = KVCacheManager(scheduler.topology, model)
    scheduler.p_chunk = 512
    scheduler._joint_horizon = CausalHorizonEstimator(
        cold_start_default_forward_steps=1)
    scheduler._joint_flows = LinkFlowRegistry()
    scheduler._pool_ports = _PoolPortRegistry()
    scheduler._hbm_ports = SimpleNamespace()  # off 臂不触端口注册表
    scheduler._joint_factors = ServiceFactors()
    scheduler._joint_rates = JointHardwareRates.from_gbps(
        noc_link_gbps=hardware.d2d_bandwidth_gbps,
        pool_port_gbps=50.0,
        local_hbm_gbps=hardware.local_hbm_bandwidth_gbps,
        d2d_latency_ns=int(hardware.d2d_latency_ns),
        pool_latency_ns=0,
    )
    scheduler._instance_edge_ports = {}
    from joint.joint_config import parse_joint_config
    scheduler.joint_config = parse_joint_config(env={})
    scheduler._joint_mode = scheduler.joint_config.scheduler_mode
    scheduler._prefill_task_cache = {}
    scheduler._decode_task_load_cache = {}
    scheduler._task_load_cache_capacity = _TASK_LOAD_CACHE_CAPACITY
    scheduler._quota_tracker = None  # off 档（parse_joint_config 缺省）
    scheduler._link_telemetry_rates = {}
    scheduler._link_telemetry_flow_counts = {}
    from joint.predictor_cost_accounting import PredictorCostAccounting
    scheduler.predictor_cost = PredictorCostAccounting()
    scheduler._telemetry_last_tick_ns = 0
    scheduler._admission_decision_wall_ns_total = 0
    scheduler._admission_decision_wall_ns_max = 0
    scheduler._admission_decision_count = 0
    scheduler._admission_premodel_wall_ns_total = 0
    scheduler._admission_enum_wall_ns_total = 0
    scheduler._quota_verdict_wall_ns_total = 0
    scheduler._quota_verdict_candidate_checks = 0
    scheduler._joint_action_selection_counts = {"stay": 0, "copy": 0}
    scheduler._snapshot_verify = False
    scheduler._kv_ledger_epoch = 0
    scheduler._ready_frontier = set()
    scheduler.online_log_rows = []
    scheduler.online_log_count = 0
    scheduler.decision_log_sink = None
    scheduler.graph = _GraphStub()
    scheduler.ledger_admitted = {}
    scheduler.ledger_issued = {}
    scheduler._emitted_by_delivery = {0: {"requests": []}}
    scheduler._batch = {
        "assignments": [], "watches": [], "delivery_sequence": 0}
    # v9（2026-10-20）：轮转游标 + 强制钩子属性（__init__ 同款初值）。
    scheduler._joint_tie_cursor = 0
    scheduler._force_boundary_request = None
    scheduler._force_boundary_action = None
    scheduler._force_entry_request = None
    scheduler._force_entry_action = None
    scheduler._force_boundary_target = None
    scheduler._force_entry_target = None
    # quota-deferred 全路径用账面（_quota_filter_candidates / 折叠计数）。
    scheduler._quota_deferred_wait_counts = {
        "capacity": 0, "quota_link": 0, "quota_port": 0}
    scheduler._admission_failure_state = {}
    scheduler._admit_attempt_epoch = {}
    scheduler.runtime_by_request_id = {}
    return scheduler


def _runtime(request_id="r_tie", session_id="s_tie", input_tokens=64):
    runtime = _OnlineRequestRuntime({
        "request_id": request_id,
        "session_id": session_id,
        "turn_index": 0,
        "queue_index": 0,
        "prefill_length": input_tokens,
        "decode_length": 4,
        "history_tokens_before": 0,
        "prefill_context_tokens": input_tokens,
        "final_context_tokens": input_tokens + 4,
    }, 512)
    runtime.estimated_arrival_ns = 0
    return runtime


def _make_fake_selector(probe, *, instances=(0, 1), stay_cost=500,
                        copy_cost=900, stay_applicable=True):
    """P 入口选择器打桩：实现真实契约（argmin + 轮转 + 诊断三键），
    记录每次调用收到的 tie_cursor（打桩的是计价遍历 native 面，选择
    语义本身钉被测接线）。"""

    def _select(*, mode, cost_model, session, request,
                prep_frontier_by_instance=None, tie_cursor=0):
        probe.append({
            "request_id": request.request_id,
            "tie_cursor": int(tie_cursor),
        })
        candidates = []
        for index in instances:
            candidates.append(_candidate(
                index, "stay", cost_ns=stay_cost,
                applicable=stay_applicable))
            candidates.append(_candidate(index, "copy", cost_ns=copy_cost))
        applicable = [
            item for item in candidates
            if item.applicable and item.cost_ns is not None]
        best = min(applicable, key=lambda item: (
            item.cost_ns, item.order_key()))
        rotated, tie_instances = joint_tie_rotation_pick(
            applicable, int(tie_cursor), sorted(instances))
        if rotated is not None:
            best = rotated
        return SelectionRecord(
            mode=mode, chosen=best, candidates=tuple(candidates),
            instance_rule_note="joint instance x action argmin",
            diagnostics={
                "decision_stage": "P",
                "joint_tie_cursor": int(tie_cursor),
                "joint_tie_instances": tie_instances,
                "joint_tie_rotated": rotated is not None,
            })

    return _select


def _decision_rows(scheduler, kind):
    return [row for row in scheduler.online_log_rows
            if row["kind"] == kind]


# ============================================ 1. 配额重选轮转镜像 ==

class QuotaReselectRotationMirrorTest(unittest.TestCase):
    """_quota_reselect_feasible joint 分支 = 主臂同判据含轮转（v9）。"""

    def test_joint_branch_rotates_over_shrunk_feasible_set(self):
        """可行集缩集（实例 {1,3} 并列最优）+ 全集 {0,1,2,3}：游标 0
        取 1（argmin 同旧序）；游标 2 沿全集环序自 2 起取首个在集
        并列实例 = 3（跳过被配额剔除的 2）；游标 3 取 3（自身在集）；
        游标 4 越全集最大值回绕到 1。"""
        scheduler = _bare_scheduler(mode="joint", cursor=0)
        cost_model = SimpleNamespace(loads=_load_views(0, 1, 2, 3))
        feasible = [
            _candidate(1, "stay", cost_ns=500),
            _candidate(3, "stay", cost_ns=500),
        ]
        # 可行集外更贵候选不参与（镜像主臂 argmin 面）。
        session_view = SimpleNamespace(
            home_instance=None, resident_instance=None)
        self.assertEqual(
            scheduler._quota_reselect_feasible(
                feasible, session_view, cost_model).instance_index, 1)
        scheduler._joint_tie_cursor = 2
        self.assertEqual(
            scheduler._quota_reselect_feasible(
                feasible, session_view, cost_model).instance_index, 3)
        scheduler._joint_tie_cursor = 3
        self.assertEqual(
            scheduler._quota_reselect_feasible(
                feasible, session_view, cost_model).instance_index, 3)
        scheduler._joint_tie_cursor = 4
        self.assertEqual(
            scheduler._quota_reselect_feasible(
                feasible, session_view, cost_model).instance_index, 1)

    def test_stand_in_default_arm_mirrors_without_cost_model(self):
        """cost_model=None 替身缺省臂同构：全集退可行集实例集
        （{1,3}），游标 2 起环序取 3；游标 3 取 3（自身在集）；游标
        越最大值回绕取 1。"""
        scheduler = _bare_scheduler(mode="joint", cursor=2)
        feasible = [
            _candidate(1, "stay", cost_ns=500),
            _candidate(3, "stay", cost_ns=500),
        ]
        session_view = SimpleNamespace(
            home_instance=None, resident_instance=None)
        self.assertEqual(
            scheduler._quota_reselect_feasible(
                feasible, session_view, None).instance_index, 3)
        scheduler._joint_tie_cursor = 3
        self.assertEqual(
            scheduler._quota_reselect_feasible(
                feasible, session_view, None).instance_index, 3)
        scheduler._joint_tie_cursor = 4
        self.assertEqual(
            scheduler._quota_reselect_feasible(
                feasible, session_view, None).instance_index, 1)

    def test_single_instance_action_tie_keeps_argmin(self):
        """并列只在同实例两动作间（stay/copy 同价）= 非放置争用不轮转
        ——游标任意值下恒 argmin（ACTION_ORDER 终裁 stay）。"""
        scheduler = _bare_scheduler(mode="joint", cursor=1)
        cost_model = SimpleNamespace(loads=_load_views(0, 1))
        feasible = [
            _candidate(1, "stay", cost_ns=500),
            _candidate(1, "copy", cost_ns=500),
        ]
        session_view = SimpleNamespace(
            home_instance=None, resident_instance=None)
        chosen = scheduler._quota_reselect_feasible(
            feasible, session_view, cost_model)
        self.assertEqual(chosen.instance_index, 1)
        self.assertEqual(chosen.action, "stay")

    def test_sequential_branch_does_not_rotate(self):
        """load-first 分支不受游标影响：可行集上按负载选实例（实例 1
        负载最低——0/2 排队 100ns），跨实例同价平局仍固定确定序——
        游标 0/1/2 选择不变。"""
        loads = _load_views(0, 1, 2)
        loads[0] = InstanceLoadView(
            instance_index=0,
            queued_task_load_ns=100, running_task_load_ns=0,
            active_decode_task_load_ns=0,
            hbm_remaining_bytes_by_tp_rank=(1, 1),
            reclaimable_bytes_by_tp_rank=(0, 0))
        loads[2] = InstanceLoadView(
            instance_index=2,
            queued_task_load_ns=100, running_task_load_ns=0,
            active_decode_task_load_ns=0,
            hbm_remaining_bytes_by_tp_rank=(1, 1),
            reclaimable_bytes_by_tp_rank=(0, 0))
        cost_model = SimpleNamespace(loads=loads)
        session_view = SimpleNamespace(
            home_instance=None, resident_instance=None)
        feasible = [
            _candidate(0, "stay", cost_ns=500),
            _candidate(1, "stay", cost_ns=500),
            _candidate(2, "stay", cost_ns=500),
        ]
        for cursor in (0, 1, 2):
            scheduler = _bare_scheduler(
                mode=SCHEDULER_MODE_LOAD_FIRST, cursor=cursor)
            self.assertEqual(
                scheduler._quota_reselect_feasible(
                    feasible, session_view, cost_model).instance_index, 1,
                cursor)


# ============================================ 2. 推进判据 ==

class ConsumeTieRotationTest(unittest.TestCase):
    """_consume_joint_tie_rotation：游标唯一推进点的判据钉死。"""

    def test_joint_multicast_tie_advances_to_next_instance(self):
        """joint 模式跨实例平局（stay@0/1/2 并列最优、选中 1）⇒ 游标
        推进到环序下一元素 2；选中 2（最大编号）⇒ 回绕到 0。"""
        scheduler = _bare_scheduler(mode="joint", cursor=0)
        scheduler._consume_joint_tie_rotation(
            _tie_record(chosen_instance=1), forced=False)
        self.assertEqual(scheduler._joint_tie_cursor, 2)
        scheduler._consume_joint_tie_rotation(
            _tie_record(chosen_instance=2), forced=False)
        self.assertEqual(scheduler._joint_tie_cursor, 0)

    def test_non_joint_modes_do_not_advance(self):
        """顺序参照（load-first）与对照臂（face_static）不轮转——
        固定确定序，游标不动。"""
        for mode in (SCHEDULER_MODE_LOAD_FIRST,
                     SCHEDULER_MODE_FACE_STATIC):
            scheduler = _bare_scheduler(mode=mode, cursor=0)
            scheduler._consume_joint_tie_rotation(
                _tie_record(chosen_instance=1, mode=mode), forced=False)
            self.assertEqual(
                scheduler._joint_tie_cursor, 0, mode)

    def test_forced_hit_does_not_advance(self):
        """probe 强制钩子命中（forced=True）绕过轮转裁决——不消费
        游标（否则一次强制污染后续自然选择的分摊起点）。"""
        scheduler = _bare_scheduler(mode="joint", cursor=0)
        scheduler._consume_joint_tie_rotation(
            _tie_record(chosen_instance=1), forced=True)
        self.assertEqual(scheduler._joint_tie_cursor, 0)

    def test_unique_best_does_not_advance(self):
        """唯一最优实例（实例 2 独占最优价）= 非平局不推进。"""
        scheduler = _bare_scheduler(mode="joint", cursor=0)
        candidates = (
            _candidate(0, "stay", cost_ns=800),
            _candidate(1, "stay", cost_ns=700),
            _candidate(2, "stay", cost_ns=500),
            _candidate(0, "copy", cost_ns=900),
            _candidate(1, "copy", cost_ns=900),
            _candidate(2, "copy", cost_ns=900),
        )
        record = SelectionRecord(
            mode="joint", chosen=candidates[2], candidates=candidates,
            instance_rule_note="joint instance x action argmin")
        scheduler._consume_joint_tie_rotation(record, forced=False)
        self.assertEqual(scheduler._joint_tie_cursor, 0)

    def test_chosen_off_tie_face_does_not_advance(self):
        """选择被顶到并列面外（chosen 比最优适用价贵——强制形态）⇒
        选择未消费轮转，不推进。"""
        scheduler = _bare_scheduler(mode="joint", cursor=0)
        scheduler._consume_joint_tie_rotation(
            _tie_record(chosen_instance=None), forced=False)
        # chosen=None ⇒ argmin = stay@0（在并列面上）——先证对照推进。
        self.assertEqual(scheduler._joint_tie_cursor, 1)
        scheduler._joint_tie_cursor = 0
        candidates = (
            _candidate(0, "stay", cost_ns=500),
            _candidate(1, "stay", cost_ns=500),
            _candidate(0, "copy", cost_ns=900),
            _candidate(1, "copy", cost_ns=900),
        )
        off_face = SelectionRecord(
            mode="joint", chosen=candidates[2], candidates=candidates,
            instance_rule_note="forced")
        scheduler._consume_joint_tie_rotation(off_face, forced=True)
        self.assertEqual(scheduler._joint_tie_cursor, 0)

    def test_single_instance_action_tie_does_not_advance(self):
        """并列只在同实例两动作间（stay/copy 同价）⇒ 非放置争用，
        不推进（与旧确定序行为恒等）。"""
        scheduler = _bare_scheduler(mode="joint", cursor=0)
        candidates = (
            _candidate(0, "stay", cost_ns=500),
            _candidate(0, "copy", cost_ns=500),
            _candidate(1, "stay", cost_ns=900),
            _candidate(1, "copy", cost_ns=900),
        )
        record = SelectionRecord(
            mode="joint", chosen=candidates[0], candidates=candidates,
            instance_rule_note="joint instance x action argmin")
        scheduler._consume_joint_tie_rotation(record, forced=False)
        self.assertEqual(scheduler._joint_tie_cursor, 0)


# ============================================ 3. 全路径（P 准入）==

class AdmissionFullPathTieWiringTest(unittest.TestCase):
    """真实 _try_admit_request（选择器打桩）：三键披露 + 推进时序。"""

    def test_success_rows_carry_captured_cursor_and_advance(self):
        """连续两次准入：行三键 = 决策起点捕获值（非落盘时成员现值），
        选择器收到的 tie_cursor 同刻；每次有效选择后游标前移（0→1→
        回绕 0），第二行选中实例 1（轮转分摊可见）。"""
        scheduler = _full_scheduler()
        probe = []
        with mock.patch.object(
                _sh30_module, "select_instance_and_action",
                new=_make_fake_selector(probe)):
            runtime_a = _runtime("r_a", session_id="s_a")
            self.assertTrue(scheduler._try_admit_request(runtime_a, 0))
            self.assertEqual(scheduler._joint_tie_cursor, 1)
            runtime_b = _runtime("r_b", session_id="s_b")
            self.assertTrue(scheduler._try_admit_request(runtime_b, 0))
            # 第二次推进自实例 1 越最大值回绕到 0。
            self.assertEqual(scheduler._joint_tie_cursor, 0)
        self.assertEqual([call["tie_cursor"] for call in probe], [0, 1])
        rows = _decision_rows(scheduler, "joint_admission")
        self.assertEqual(len(rows), 2)
        first, second = (row["decision"] for row in rows)
        self.assertEqual(first["joint_tie_cursor"], 0)
        self.assertEqual(first["joint_tie_instances"], [0, 1])
        self.assertTrue(first["joint_tie_rotated"])
        self.assertEqual(first["joint_instance_index"], 0)
        self.assertEqual(second["joint_tie_cursor"], 1)
        self.assertEqual(second["joint_instance_index"], 1)
        self.assertTrue(second["joint_tie_rotated"])
        # 成员现值（已回绕到 0）≠ 第二行披露的捕获值 1——捕获口径钉死。
        self.assertNotEqual(
            second["joint_tie_cursor"], scheduler._joint_tie_cursor)

    def test_capacity_failure_advances_and_discloses_captured_cursor(self):
        """KVCapacityError 路径：游标照推（容量失败重试需新游标分摊），
        失败首行（joint_admission_failed）披露决策起点捕获游标（推进
        已发生，成员现值 ≠ 披露值）；同请求第二次失败折叠 wait 行
        （joint_admission_wait）无三键。"""
        from face_scheduler import KVCapacityError
        scheduler = _full_scheduler()
        probe = []
        with mock.patch.object(
                _sh30_module, "select_instance_and_action",
                new=_make_fake_selector(probe)), \
            mock.patch.object(
                scheduler.kv_manager, "reserve_request_capacity",
                side_effect=KVCapacityError("injected capacity failure")):
            runtime = _runtime("r_cap")
            self.assertFalse(scheduler._try_admit_request(runtime, 0))
            self.assertEqual(scheduler._joint_tie_cursor, 1)
            # 同请求第二次容量失败（纪元门在测试直调下不复用——直接
            # 再调一次走折叠分支）。
            self.assertFalse(scheduler._try_admit_request(runtime, 0))
        self.assertEqual([call["tie_cursor"] for call in probe], [0, 1])
        failed = _decision_rows(scheduler, "joint_admission_failed")
        wait = _decision_rows(scheduler, "joint_admission_wait")
        self.assertEqual(len(failed), 1)
        self.assertEqual(len(wait), 1)
        decision = failed[0]["decision"]
        self.assertEqual(decision["failure_class"], "capacity_deferred")
        self.assertEqual(decision["joint_tie_cursor"], 0)
        self.assertEqual(decision["joint_tie_instances"], [0, 1])
        self.assertTrue(decision["joint_tie_rotated"])
        # 折叠 wait 行不带三键。
        for key in ("joint_tie_cursor", "joint_tie_instances",
                    "joint_tie_rotated"):
            self.assertNotIn(key, wait[0]["decision"])

    def test_quota_deferred_neither_advances_nor_discloses(self):
        """quota-deferred 提前 return 在强制钩子/推进点之前：游标不
        动、选择器仅被调一次；失败首行（quota_deferred 类）无三键
        （调用点不传捕获值——等待折叠族口径）。"""
        scheduler = _full_scheduler()
        scheduler._quota_tracker = LinkQuotaTracker(
            mode="static",
            noc_link_bytes_per_ns=(
                scheduler._joint_rates.noc_link_bytes_per_ns),
            local_hbm_bytes_per_ns=(
                scheduler._joint_rates.local_hbm_bytes_per_ns),
            delta_adm_ns=0,
        )
        blocked = QuotaVerdict(
            admitted=False, resource_kind="link", wait_reason="quota_link",
            inapplicable_reason="quota_link: injected blocked verdict")
        probe = []
        with mock.patch.object(
                _sh30_module, "select_instance_and_action",
                new=_make_fake_selector(probe)), \
            mock.patch.object(
                scheduler, "_quota_candidate_verdict",
                return_value=blocked):
            runtime = _runtime("r_qd")
            self.assertFalse(scheduler._try_admit_request(runtime, 0))
        self.assertEqual(scheduler._joint_tie_cursor, 0)
        self.assertEqual(len(probe), 1)
        failed = _decision_rows(scheduler, "joint_admission_failed")
        wait = _decision_rows(scheduler, "joint_admission_wait")
        rows = failed + wait
        self.assertEqual(len(rows), 1)
        for key in ("joint_tie_cursor", "joint_tie_instances",
                    "joint_tie_rotated"):
            self.assertNotIn(key, rows[0]["decision"])


# ============================================ 4. L2 键含游标 ==

class ArrivalSelectionCacheKeyTest(unittest.TestCase):
    """_joint_arrival_selection L2 键并入游标分量：推进后必 miss。"""

    def _cache_scheduler(self):
        scheduler = Sh30OnlineScheduler.__new__(Sh30OnlineScheduler)
        scheduler._joint_mode = "joint"
        scheduler._joint_tie_cursor = 0
        scheduler._joint_arrival_selection_cache = {}
        scheduler._joint_arrival_cache_stats = {
            "l1_hits": 0, "l1_misses": 0,
            "l2_hits": 0, "l2_misses": 0,
            "verify_checks": 0, "verify_mismatches": 0,
        }
        scheduler._joint_arrival_cache_verify = False
        scheduler.instances = [
            _OnlineInstanceState(index=index) for index in range(2)]
        scheduler._joint_horizon = CausalHorizonEstimator(
            cold_start_default_forward_steps=1)
        scheduler._joint_prediction_config = SimpleNamespace(
            future_demand_mode="none")
        scheduler._joint_flows = SimpleNamespace(state_version=0)
        scheduler._joint_factors = SimpleNamespace(service_version=0)
        return scheduler

    @staticmethod
    def _views():
        session_view = SessionKVView(
            session_id="s_l2", home_instance=None, resident_instance=None,
            location="none", history_tokens=0, resident_prefix_layers=0,
            history_bytes_by_tp_rank=(0, 0),
            missing_bytes_by_tp_rank=(0, 0))
        request_view = RequestView(
            request_id="r_l2", session_id="s_l2", input_tokens=64,
            history_tokens_before=0, estimated_decode_tokens=4,
            horizon_source="cold_start_default",
            input_kv_bytes_by_tp_rank=(128, 128))
        return session_view, request_view

    def test_cursor_advance_forces_l2_miss_and_reselect(self):
        """同 request_view 同快照：首调 miss 入缓存、二调命中（选择器
        不再被调）；游标推进后键变化 ⇒ 必 miss 重算（选择器以新游标
        值再被调，l2_misses +1）——旧游标下的 SelectionRecord 不复用。"""
        scheduler = self._cache_scheduler()
        session_view, request_view = self._views()
        probe = []
        fake = _make_fake_selector(probe)
        with mock.patch.object(
                _sh30_module, "select_instance_and_action", new=fake):
            first = scheduler._joint_arrival_selection(
                0, SimpleNamespace(), session_view, request_view)
            self.assertEqual(len(probe), 1)
            self.assertEqual(probe[0]["tie_cursor"], 0)
            stats_after_first = dict(scheduler._joint_arrival_cache_stats)
            self.assertEqual(stats_after_first["l2_misses"], 1)
            self.assertEqual(stats_after_first["l2_hits"], 0)
            cached = scheduler._joint_arrival_selection(
                0, SimpleNamespace(), session_view, request_view)
            self.assertIs(cached, first)
            self.assertEqual(len(probe), 1)  # 命中不复算
            self.assertEqual(
                scheduler._joint_arrival_cache_stats["l2_hits"], 1)
            # 游标推进（模拟容量失败重试前 _consume 已消费）。
            scheduler._joint_tie_cursor = 1
            rotated = scheduler._joint_arrival_selection(
                0, SimpleNamespace(), session_view, request_view)
            self.assertEqual(len(probe), 2)  # miss 重算
            self.assertEqual(probe[1]["tie_cursor"], 1)
            self.assertEqual(
                scheduler._joint_arrival_cache_stats["l2_misses"], 2)
            # 行为差异：轮转改变了并列归属（0 → 1）——旧记录不可复用
            # 的语义证据。
            self.assertEqual(first.chosen.instance_index, 0)
            self.assertEqual(rotated.chosen.instance_index, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
