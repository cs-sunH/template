#!/usr/bin/env python3
"""test_joint_quota_integration.py -- C11（WP3c，2026-09-22）SH 配额集成的
零后端单测。

覆盖（任务卡 C11 步骤 1/5/6/7/8 + riders）：

1. **候选级配额判据**（步骤 1）：逐候选执行 link_quota 判据——链路门
   （quota_link 分列）/ 端口平价门与 merge bulk 余量（quota_port 分列）；
   动作级不适用（实例永不掩码——stay/recompute 恒适用、被判据拒绝的
   仅该动作）；配额可行集内重选 ≡ argmin（δ_adm = 0 合取恒等，A3'）。
2. **quota_deferred 闭合**（C9 语义）：全动作不可行 ⇒ 回队（永不
   JointSchedulerError）、重试键扩展配额代数、信用释放 bump 代数后
   重试门重开；等待日志 wait_reason 分列。
3. **流生命周期借还配对**（步骤 7 + 守恒）：remote-read 读流 + merge
   预留（service_done 方向裁决释放败者侧 / merge_done 释放胜者侧，
   与 kv_delta 结算闭合门同序断言）、零传输分支整体释放、copy/evict
   oneshot 同生命周期；收尾守恒审计 fail-closed。
4. **AIMD 闭环**（步骤 8）：遥测有效速率（A5'/B3 端点键换算后）喂
   observe_telemetry——收缩经 set_link_quota 落地并 bump 配额代数。
5. **port_snapshot 接通**（步骤 6）：配额 on = 实测值（C2 注册表 u_port
   分解 + tracker bulk/平价门余量）；off = C5 冻结 NA 占位。
6. **A5'/B3 rider**：LinkId → (src, dst) 端点键换算（C++ 枚举配方）、
   未知 id fail-closed、无属性替身退化透传。
7. **F7 耦合规则**：joint_runner --quota aimd ⇒ SH_LINK_TELEMETRY=1 自动
   注入 + invocation.json 记录；run_online_strategy.sh 的防御性断言
   （aimd + 无遥测 ⇒ fail-closed；显式 =1 ⇒ 追加 --link-telemetry）。

运行：cd sh_test_mesh/workload/llama2_7b_inference &&
      python3 online/test_joint_quota_integration.py   （或 pytest 同路径）
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from collections import deque
from types import SimpleNamespace

_ONLINE_DIR = os.path.dirname(os.path.abspath(__file__))
_WORKLOAD_DIR = os.path.dirname(_ONLINE_DIR)
_RUN_SCRIPTS_DIR = os.path.join(
    os.path.dirname(_WORKLOAD_DIR), os.pardir, "run_scripts")
for _p in (_ONLINE_DIR, _WORKLOAD_DIR, os.path.abspath(_RUN_SCRIPTS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from face_scheduler import (  # noqa: E402
    FaceHardware,
    FaceInstanceSpec,
    FaceModel,
    build_instances,
)
from joint.hbm_port_flow_registry import HbmPortFlowRegistry  # noqa: E402
from joint.joint_config import JointMechanismConfig  # noqa: E402
from joint.joint_cost_model import (  # noqa: E402
    ActionCandidate,
    InstanceLoadView,
    JointHardwareRates,
    ServiceFactors,  # F6 销账：替身对齐 __init__ 初值
    SessionKVView,
)
from joint.joint_scheduler import SelectionRecord  # noqa: E402
from joint.link_quota import (  # noqa: E402
    QUOTA_AIMD,
    QUOTA_STATIC,
    LinkQuotaTracker,
    challenger_flips,
)
from online.sh30_online_scheduler import (  # noqa: E402
    _OnlineInstanceState,
    _OnlineRequestRuntime,
    _PoolPortRegistry,
    _TASK_LOAD_CACHE_CAPACITY,  # F6 销账：替身对齐 __init__ 初值
    Sh30OnlineScheduler,
)
from online.online_scheduler_base import OnlineSchedulerBase  # noqa: E402


# ============================================================ 夹具构造 ==

def _hardware():
    return FaceHardware(
        mesh_rows=2,
        mesh_cols=4,
        local_hbm_capacity_bytes=1_000_000_000,
        local_hbm_bandwidth_gbps=100.0,
        d2d_bandwidth_gbps=200.0,
        peak_perf_tflops=1.0,
        d2d_latency_ns=0,
        local_hbm_latency_ns=0,
    )


def _model():
    return FaceModel(
        layers=4,
        hidden_size=64,
        ffn_size=128,
        num_heads=4,
        vocab_size=32,
        bytes_per_elem=2,
        mlp_variant="swiglu",
    )


def _topology():
    # 2×4 mesh 四实例（tp=2）：实例 0 = ranks (0,1)、实例 1 = ranks
    # (2,3)、实例 2/3 = (4,5)/(6,7)（与 C8 夹具同款——XY 路由 1→0 的
    # shard 路径 (2,1,0)/(3,2,1) 共享链路 (2,1)）。
    return build_instances(
        _hardware(),
        (FaceInstanceSpec("g0", "pg0", (0, 1)),
         FaceInstanceSpec("g1", "pg1", (2, 3)),
         FaceInstanceSpec("g2", "pg2", (4, 5)),
         FaceInstanceSpec("g3", "pg3", (6, 7))),
        require_equal_size=True)


def _rates():
    return JointHardwareRates.from_gbps(
        noc_link_gbps=200.0, pool_port_gbps=1.0,
        local_hbm_gbps=100.0, d2d_latency_ns=0, pool_latency_ns=0)


def _session_view(home=1, resident=1, history=100):
    return SessionKVView(
        session_id="s1", home_instance=home, resident_instance=resident,
        location="local_hbm", history_tokens=history,
        resident_prefix_layers=4,
        history_bytes_by_tp_rank=(400, 400),
        missing_bytes_by_tp_rank=(0, 0))


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


def _make_runtime(request_id, *, session_id="s1"):
    return _OnlineRequestRuntime({
        "request_id": request_id,
        "session_id": session_id,
        "turn_index": 0,
        "queue_index": 0,
        "prefill_length": 50,
        "decode_length": 16,
        "history_tokens_before": 100,
        "prefill_context_tokens": 150,
        "final_context_tokens": 166,
    })


def _scheduler(*, quota_mode="static"):
    """__new__ 范式（C8/C14 同款）：只装配被测路径用到的属性。

    硬件速率：B_link = 200 B/ns、B_HBM = 100 B/ns ⇒ rho_eff = 2 ⇒
    Q_init = N_bulk = 2（F2 派生结果，测试判据的可复算锚）。"""
    scheduler = Sh30OnlineScheduler.__new__(Sh30OnlineScheduler)
    scheduler.hardware = _hardware()
    scheduler.model = _model()
    scheduler.topology = _topology()
    scheduler._joint_rates = _rates()
    scheduler._decode_task_load_cache = {}
    # 对齐 __init__ 初值（F6 销账：软门已删，替身漏设 = AttributeError）。
    scheduler._prefill_task_cache = {}
    scheduler._task_load_cache_capacity = _TASK_LOAD_CACHE_CAPACITY
    scheduler._joint_factors = ServiceFactors()
    scheduler.instances = [_OnlineInstanceState(index=i)
                           for i in range(4)]
    scheduler._rank_to_instance = {
        rank: instance.index
        for instance in scheduler.topology.instances
        for rank in instance.ranks}
    scheduler.runtime_by_request_id = {}
    scheduler.kv_manager = SimpleNamespace(
        tp_degree=2,
        _sessions={},
    )
    scheduler.joint_config = JointMechanismConfig(
        category_mode="typed", scheduler_mode="joint",
        layer_policy="adaptive", remote_actions="on",
        quota_mode=quota_mode)
    scheduler._kv_ledger_epoch = 0
    scheduler._admit_attempt_epoch = {}
    scheduler._admission_failure_state = {}
    scheduler._admit_gate_verify = False
    scheduler.pending_admissions = deque()
    scheduler._joint_flows = SimpleNamespace(
        has_registrations=False, collective_coverage=False)
    scheduler._pool_ports = _PoolPortRegistry()
    scheduler._hbm_ports = HbmPortFlowRegistry()
    scheduler._hbm_ports.attach_active_decode_provider(
        scheduler._hbm_active_decode_streams)
    # C11 状态（__init__ 同款初值——真实构造在 _joint_rates 之后）。
    scheduler._quota_tracker = (
        LinkQuotaTracker(
            mode=quota_mode,
            noc_link_bytes_per_ns=(
                scheduler._joint_rates.noc_link_bytes_per_ns),
            local_hbm_bytes_per_ns=(
                scheduler._joint_rates.local_hbm_bytes_per_ns),
            delta_adm_ns=0,
        ) if quota_mode != "off" else None)
    scheduler._quota_enrolled = {}
    scheduler._quota_merge_reserves = {}
    scheduler._quota_merge_reserves_created = 0
    scheduler._quota_decode_owner_seq = {}
    scheduler._joint_action_selection_counts = {
        "stay": 0, "copy": 0, "remote-read": 0,
        "recompute_elected": 0,
        "recompute_forced_no_history": 0,
        "recompute_forced_quota_deferred": 0,
        "recompute_forced_evicted_permanent": 0,
    }
    scheduler._quota_deferred_wait_counts = {
        "capacity": 0, "quota_link": 0, "quota_port": 0}
    scheduler._quota_deferred_dwell_ns = []
    scheduler._admission_decision_wall_ns_total = 0
    scheduler._admission_decision_wall_ns_max = 0
    scheduler._admission_decision_count = 0
    scheduler._quota_verdict_wall_ns_total = 0
    scheduler._quota_verdict_candidate_checks = 0
    scheduler._quota_admit_events = 0
    scheduler._quota_release_events = 0
    scheduler._quota_aimd_action_counts = {}
    # C8 遥测状态（__init__ 同款；A5'/B3 换算表按生产路径装配）。
    scheduler._link_telemetry_rates = {}
    # M1：C8 遥测状态 __init__ 同款（时间加权活跃流数）。
    scheduler._link_telemetry_flow_counts = {}
    scheduler.p_chunk = 512  # N4：_joint_prefill_total_load_ns 同形切分（F6 替身补设）
    scheduler._link_telemetry_epoch_count = 0
    scheduler._link_telemetry_sample_count = 0
    scheduler._telemetry_absent_seen = False
    scheduler._telemetry_window_broken = False
    scheduler._telemetry_window_end_ns = 0
    scheduler._telemetry_last_tick_ns = 0
    scheduler._telemetry_link_id_map = None
    # log_decision 基类契约（__new__ 替身手工置初值）。
    scheduler.online_log_count = 0
    scheduler.decision_log_sink = None
    scheduler.online_log_rows = []
    return scheduler


def _decision_rows(scheduler, kind):
    return [row for row in scheduler.online_log_rows
            if row["kind"] == kind]


# ============================================ 1. 候选级判据（步骤 1）==

class CandidateVerdictTest(unittest.TestCase):
    def setUp(self) -> None:
        self.scheduler = _scheduler()
        self.tracker = self.scheduler._quota_tracker

    def test_local_actions_always_admitted(self):
        """stay/recompute 无跨实例主流程——恒适用（实例永不掩码）。"""
        for action in ("stay", "recompute"):
            verdict = self.scheduler._quota_candidate_verdict(
                _candidate(0, action), _session_view(), 50.0)
            self.assertTrue(verdict.admitted)

    def test_local_actions_admitted_under_saturated_quota(self):
        """A12' 批压力态钉死（⇒ 方向）：链路门与端口平价门双双占满
        （copy/remote-read 确被拒）时 stay/recompute 仍恒适用——结构性
        早退先于任何压力检查。空 tracker 用例无法区分"恒适用"与
        "恰好配额空闲"，本用例以压力态补证 deferred 仅在全动作不可行
        的安全网场景出现。"""
        # 链路门占满：预占 (2,3)→(0,1) 共享链路至 Q_init=2。
        self.assertTrue(self.tracker.admit_flow(
            owner="pre#copy", flow_class="oneshot",
            links=self.scheduler._quota_route_edges(1, 0),
            port_id=0).admitted)
        self.assertEqual(self.tracker.link_remaining((2, 1)), 0)
        # 端口平价门占满：port 3 两条 r̂=50 realtime 流（B=100/3 < 50）。
        for i in range(2):
            self.assertTrue(self.tracker.admit_flow(
                owner="pre#rt{}".format(i), flow_class="realtime",
                links=(), port_id=3,
                r_hat_kv_bytes_per_ns=50.0).admitted)
        # 压力真实在场：copy 与 remote-read 确被拒。
        self.assertFalse(self.scheduler._quota_candidate_verdict(
            _candidate(0, "copy"), _session_view(resident=1), 50.0
        ).admitted)
        # stay/recompute 仍恒适用。
        for action in ("stay", "recompute"):
            self.assertTrue(self.scheduler._quota_candidate_verdict(
                _candidate(0, action), _session_view(), 50.0).admitted)

    def test_link_gate_blocks_copy_action_only(self):
        """链路门拒 copy（quota_link 分列）——只该动作不适用，同实例
        其余动作不受影响（动作级、非实例掩码）。"""
        # 预占 (2,3)→(0,1) 路径的共享链路至 Q_init=2 满：实例 1→0 的
        # 两条 shard 路径 (2,1,0)/(3,2,1) 各占链路 (2,1) 一次 ⇒ 一条
        # copy 流 demand=2 恰耗尽；再判第二条即拒。
        verdict_ok = self.tracker.admit_flow(
            owner="pre#copy", flow_class="oneshot",
            links=self.scheduler._quota_route_edges(1, 0), port_id=0)
        self.assertTrue(verdict_ok.admitted)
        self.assertEqual(self.tracker.link_remaining((2, 1)), 0)
        verdict = self.scheduler._quota_candidate_verdict(
            _candidate(0, "copy"), _session_view(resident=1), 50.0)
        self.assertFalse(verdict.admitted)
        self.assertEqual(verdict.resource_kind, "link")
        self.assertEqual(verdict.wait_reason, "quota_link")
        self.assertIn("quota_link", verdict.inapplicable_reason)
        # 同实例 stay 不受影响。
        self.assertTrue(self.scheduler._quota_candidate_verdict(
            _candidate(0, "stay"), _session_view(), 50.0).admitted)

    def test_parity_gate_blocks_remote_read(self):
        """端口平价门拒 remote-read（quota_port 分列）：B_HBM/(u+1) <
        r̂（B=100，r̂=50：enrolled=2 ⇒ 100/3 < 50）。路由 1→3 的链路
        需求（读流 1 + 预留 fwd 1 = 每链恰 Q=2）不先触发链路门。"""
        for i in range(2):
            self.tracker.admit_flow(
                owner="pre#rt{}".format(i), flow_class="realtime",
                links=(), port_id=3,
                r_hat_kv_bytes_per_ns=50.0)
        verdict = self.scheduler._quota_candidate_verdict(
            _candidate(3, "remote-read"), _session_view(resident=1), 50.0)
        self.assertFalse(verdict.admitted)
        self.assertEqual(verdict.resource_kind, "port")
        self.assertEqual(verdict.wait_reason, "quota_port")
        self.assertIn("parity", verdict.inapplicable_reason)

    def test_merge_bulk_headroom_in_verdict(self):
        """remote-read 判据含双候选胜者侧 bulk 余量前瞻（N_bulk=2）。"""
        for i in range(2):
            self.tracker.reserve_merge(
                "pre#m{}".format(i),
                links_forward=(), links_reverse=(),
                port_forward=3, port_reverse=1)
        verdict = self.scheduler._quota_candidate_verdict(
            _candidate(3, "remote-read"), _session_view(resident=1), 0.01)
        self.assertFalse(verdict.admitted)
        self.assertEqual(verdict.wait_reason, "quota_port")
        self.assertIn("bulk", verdict.inapplicable_reason)


class DeltaAdmZeroConjunctionTest(unittest.TestCase):
    """δ_adm = 0（A3' 终冻）与既有 argmin 的合取恒等。"""

    def test_predicate_identity_with_argmin_order(self):
        """δ_adm = 0（A3' 终冻）翻转谓词：手工推导查表。

        F4（复审修复）：原期望式 ``incumbent - challenger >= 0`` 与
        challenger_flips 实现裁决式同源（恒真恒等，零效）——改为写死
        查表，期望值逐对手工推导：语义 = 挑战者成本不贵于在册
        （challenger <= incumbent）才允许翻转。"""
        # (challenger, incumbent, expected_flip)
        # 0 < 500：挑战者更便宜 → 允许翻转 → True
        # 1 < 500：同上 → True
        # 499 < 500：差 1 的近平局，δ=0 不设带 → True
        # 500 == 500：平局边界（δ=0 含等号，谓词开；平局终裁归
        #   argmin 的 order_key，谓词不夹带语义）→ True
        # 0 == 0：零成本平局边界 → True
        # 3286 == 3286：大值平局边界 → True
        # 10**9 == 10**9：极端值平局边界 → True
        # 500 > 0：挑战者更贵 → 拒翻转 → False
        # 501 > 500：差 1 更贵 → False
        # 3286 > 500：更贵 → False
        # 10**9 > 3286：极端更贵 → False
        table = (
            (0, 500, True),
            (1, 500, True),
            (499, 500, True),
            (500, 500, True),
            (0, 0, True),
            (3286, 3286, True),
            (10**9, 10**9, True),
            (500, 0, False),
            (501, 500, False),
            (3286, 500, False),
            (10**9, 3286, False),
        )
        for challenger, incumbent, expected in table:
            self.assertIs(
                challenger_flips(challenger, incumbent, 0), expected,
                (challenger, incumbent))

    def test_reselection_among_feasible_equals_argmin(self):
        scheduler = _scheduler()
        # 占满实例 0 的链路 ⇒ copy@0 被拒；其余候选（copy@2 更贵、
        # stay@3 更便宜）照常——重选 = 配额可行集的 argmin 同序。
        tracker = scheduler._quota_tracker
        tracker.admit_flow(
            owner="pre", flow_class="oneshot",
            links=scheduler._quota_route_edges(1, 0), port_id=0)
        record = SelectionRecord(
            mode="joint",
            chosen=_candidate(0, "copy", cost_ns=100),
            candidates=(
                _candidate(0, "stay", cost_ns=800),
                _candidate(0, "copy", cost_ns=100),
                _candidate(2, "copy", cost_ns=300),
                _candidate(3, "stay", cost_ns=200),
            ),
            instance_rule_note="joint instance x action argmin",
            remote_enabled=True)
        chosen, candidates, deferred = scheduler._quota_filter_candidates(
            record, _session_view(resident=1), "r1", 0)
        self.assertIsNone(deferred)
        self.assertEqual(chosen.action, "stay")
        self.assertEqual(chosen.instance_index, 3)
        blocked = [c for c in candidates
                   if c.instance_index == 0 and c.action == "copy"]
        self.assertEqual(len(blocked), 1)
        self.assertFalse(blocked[0].applicable)
        # M2（2026-09-23 验收审计）：配额拒保留配额前成本——离线
        # 双域可从决策日志重建（修前清 None 钉的是旧语义）。
        self.assertEqual(blocked[0].cost_ns, 100)
        self.assertIsNone(blocked[0].breakdown)
        self.assertIn("quota_link", blocked[0].inapplicable_reason)


# ================================ 2. quota_deferred 闭合（C9 语义）==

class QuotaDeferredClosureTest(unittest.TestCase):
    def setUp(self) -> None:
        self.scheduler = _scheduler()
        self.tracker = self.scheduler._quota_tracker

    def _all_blocked_record(self):
        # 只有 remote-read@3 预适用（其余预不适用），端口平价门拒之
        # （路由 1→3 链路需求恰 Q，不先触发链路门；内部派生 r̂ ≈
        # 23.3 B/ns ⇒ enrolled=4 时 100/5 < r̂ 拒之）。
        for i in range(4):
            self.tracker.admit_flow(
                owner="pre#rt{}".format(i), flow_class="realtime",
                links=(), port_id=3,
                r_hat_kv_bytes_per_ns=20.0)
        return SelectionRecord(
            mode="joint",
            chosen=_candidate(3, "remote-read", cost_ns=100),
            candidates=(
                _candidate(3, "stay", applicable=False, cost_ns=None),
                _candidate(3, "recompute", applicable=False, cost_ns=None),
                _candidate(3, "copy", applicable=False, cost_ns=None),
                _candidate(3, "remote-read", cost_ns=100),
            ),
            instance_rule_note="joint instance x action argmin",
            remote_enabled=True)

    def test_all_blocked_requeues_without_fail_closed(self):
        """全动作配额不可行 ⇒ 回队（不 raise JointSchedulerError），
        wait_reason = quota_port、重试键扩展配额代数。"""
        chosen, _candidates, deferred = (
            self.scheduler._quota_filter_candidates(
                self._all_blocked_record(), _session_view(resident=1),
                "r1", 0))
        self.assertIsNone(chosen)
        record, _r_hat = deferred
        self.assertTrue(record.requeue)
        self.assertEqual(record.wait_reason, "quota_port")
        self.assertEqual(record.retry_key[-1],
                         self.tracker.quota_retry_key()[0])

    def test_retry_gate_reopens_on_quota_epoch_bump(self):
        """信用释放 bump 配额代数 ⇒ _current_retry_key 与失败键分离
        （deferred 请求必获再评估）。"""
        self.tracker.admit_flow(
            owner="pre#rt0", flow_class="realtime",
            links=(), port_id=3, r_hat_kv_bytes_per_ns=50.0)
        self.scheduler._last_admit_failure_key = (
            self.scheduler._compose_admit_failure_key({0}))
        last_key = self.scheduler._last_admit_failure_key
        self.assertEqual(
            last_key, self.scheduler._current_retry_key(last_key))
        self.tracker.release_flow("pre#rt0", now_ns=10)
        self.assertNotEqual(
            last_key, self.scheduler._current_retry_key(last_key))

    def test_off_mode_retry_key_shape_unchanged(self):
        """off（tracker = None）失败键 = 既有纯容量形态（零漂移）。"""
        self.scheduler._quota_tracker = None
        key = self.scheduler._compose_admit_failure_key({0})
        self.assertEqual(key[0], 0)
        self.assertEqual(key[1], ((0, 0),))
        self.assertEqual(
            key, self.scheduler._current_retry_key(key))

    def test_wait_reason_log_columns(self):
        """joint_admission_wait 行携带 wait_reason 分列（折叠分支）。"""
        self.scheduler._admission_failure_state["r1"] = {
            "class": "quota_deferred_quota_link", "count": 1}
        self.scheduler._log_admission_failure(
            _make_runtime("r1"), 500, None,
            reason="quota_link: link=(2, 1) remaining=0",
            failure_class="quota_deferred_quota_link")
        rows = _decision_rows(self.scheduler, "joint_admission_wait")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["decision"]["wait_reason"], "quota_link")
        self.assertEqual(
            rows[0]["decision"]["failure_class"],
            "quota_deferred_quota_link")


# ====================== 3. 流生命周期借还配对（步骤 7 + 守恒）==

class QuotaLifecyclePairingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.scheduler = _scheduler()
        self.tracker = self.scheduler._quota_tracker

    def test_remote_read_full_lifecycle(self):
        """读流入册（双端点 realtime）+ merge 预留（双向各 1 槽 +
        双候选 bulk）→ service_done 裁决释放败者侧 → merge_done 释放
        胜者侧 → 守恒清零。释放时序与 C14 闭合门同序（kv_delta 行
        在案先于胜者侧释放）由 _on_merge_done 集成路径钉死（本测试
        直接驱动 _quota_* 半）。"""
        runtime = _make_runtime("r1")
        session_view = _session_view(home=1, resident=1)
        self.assertTrue(self.scheduler._quota_enroll_admission(
            runtime, session_view, "remote-read", 3, now_ns=100))
        # 双端点在册：实例 3（写/消费腿）与实例 1（源读腿）各 1 条。
        self.assertEqual(self.tracker.port_enrolled(3), 1)
        self.assertEqual(self.tracker.port_enrolled(1), 1)
        self.assertIn("r1#readplan", self.scheduler._quota_enrolled)
        self.assertIn("r1#readplan#src", self.scheduler._quota_enrolled)
        # merge 预留：双向链路各 1 槽 + 双候选 bulk 名额。
        reserve = self.scheduler._quota_merge_reserves["r1"]
        self.assertEqual(reserve["state"], "reserved")
        self.assertEqual(self.tracker.bulk_used(3), 1)
        self.assertEqual(self.tracker.bulk_used(1), 1)
        # service_done（胜者 = 执行端 3）：败者侧（home 方向）释放。
        runtime.merge_outcome = {
            "direction": "forward", "winner_instance": 3,
            "loser_instance": 1}
        runtime.merge_transfers = ("nonempty",)
        self.scheduler._quota_on_service_done("r1", runtime, 900)
        self.assertEqual(reserve["state"], "adjudicated")
        # 胜者侧 bulk 保持、读流仍在册（完成边界才释放）。
        self.assertEqual(self.tracker.bulk_used(3), 1)
        self.assertEqual(self.tracker.bulk_used(1), 0)
        # 完成边界：读流 settle（流寿命样本进 EWMA 通道）。
        self.scheduler._quota_release_readplan_stream("r1", 950)
        self.assertNotIn("r1#readplan", self.scheduler._quota_enrolled)
        self.assertNotIn("r1#readplan#src", self.scheduler._quota_enrolled)
        # merge_done：胜者侧释放 → 守恒清零。
        self.scheduler._quota_release_merge_reserve("r1")
        self.assertEqual(self.tracker.bulk_used(3), 0)
        self.assertEqual(self.scheduler._quota_merge_reserves, {})
        self.assertEqual(self.scheduler._quota_enrolled, {})

    def test_zero_transfer_branch_releases_both_sides(self):
        """零传输分支（in_place / 零字节翻转 ⇒ 无 merge watch）在
        service_done 整体释放，不等待不存在的 merge_done。"""
        runtime = _make_runtime("r2")
        self.assertTrue(self.scheduler._quota_enroll_admission(
            runtime, _session_view(home=1, resident=1), "remote-read",
            3, now_ns=100))
        runtime.merge_outcome = {
            "direction": "in_place", "winner_instance": 3,
            "loser_instance": None}
        runtime.merge_transfers = ()
        self.scheduler._quota_on_service_done("r2", runtime, 900)
        self.assertEqual(self.scheduler._quota_merge_reserves, {})
        self.assertEqual(self.tracker.bulk_used(3), 0)

    def test_unadjudicated_merge_done_fails_closed(self):
        """merge_done 到达而预留未裁决 = watch 通道与账目脱钩，
        fail-closed。"""
        runtime = _make_runtime("r3")
        self.assertTrue(self.scheduler._quota_enroll_admission(
            runtime, _session_view(home=1, resident=1), "remote-read",
            3, now_ns=100))
        with self.assertRaises(RuntimeError) as ctx:
            self.scheduler._quota_release_merge_reserve("r3")
        self.assertIn("never adjudicated", str(ctx.exception))

    def test_copy_and_eviction_oneshot_lifecycle(self):
        """copy 主流程（双端点 oneshot）+ 准入逐出支链：drain 边界
        settle。"""
        runtime = _make_runtime("r4")
        runtime.history_evictions = ()
        self.assertTrue(self.scheduler._quota_enroll_admission(
            runtime, _session_view(home=1, resident=1), "copy",
            0, now_ns=100))
        self.assertIn("r4", self.scheduler._quota_enrolled)
        self.assertIn("r4#src", self.scheduler._quota_enrolled)
        self.scheduler._quota_release_admission_phase("r4", 300)
        self.assertEqual(self.scheduler._quota_enrolled, {})

    def test_decode_phase_enrollment_sequence(self):
        """decode 相增量流各自独立 owner（#decode#q{seq}），完成边界
        全部 settle。"""
        self.scheduler._quota_enroll_decode_phase(
            "r5", (), 100)  # 空传输集 = 无入册
        self.assertEqual(self.scheduler._quota_enrolled, {})
        edges = self.scheduler._quota_route_edges(1, 0)
        from face_scheduler import KVTransfer, KVTransferShard
        transfer = KVTransfer(
            kind="noc_migrate", phase="decode", reason="test",
            session_id="s1", trigger_request_id="r5",
            source_instance_index=1, target_instance_index=0,
            total_bytes=8,
            shards=(KVTransferShard(
                source_rank=2, target_rank=0, edge_rank=None, bytes=8,
                noc_path=(2, 1, 0), layer_start=0, layer_end=4),),
            model_layers=4, layer_start=0, layer_end=4,
            resident_prefix_layers_before=4,
            resident_prefix_layers_after=4)
        self.scheduler._quota_enroll_decode_phase("r5", (transfer,), 100)
        self.scheduler._quota_enroll_decode_phase("r5", (transfer,), 200)
        self.assertEqual(
            sorted(self.scheduler._quota_enrolled),
            ["r5#decode#q0", "r5#decode#q1"])
        self.scheduler._quota_release_decode_phase("r5", 300)
        self.assertEqual(self.scheduler._quota_enrolled, {})


# ================================== 4. AIMD 闭环（步骤 8）==

class AimdClosedLoopTest(unittest.TestCase):
    def setUp(self) -> None:
        self.scheduler = _scheduler(quota_mode="aimd")
        self.tracker = self.scheduler._quota_tracker

    def _enroll_flow_on_link(self):
        self.tracker.admit_flow(
            owner="live#readplan", flow_class="realtime",
            links=((3, 2),), port_id=0,
            r_hat_kv_bytes_per_ns=0.001, now_ns=0)
        # 在册校验：link (3,2) 占用 1（每流换算分母）。
        self.assertEqual(self.tracker.link_occupancy((3, 2)), 1)

    def test_shrink_via_telemetry_bumps_quota_epoch(self):
        """端点键遥测（rate < r̂_KV）⇒ observe_telemetry 收缩经
        set_link_quota 落地、bump 配额代数（deferred 重试门重开）。"""
        self._enroll_flow_on_link()
        r_hat = self.scheduler._quota_r_hat_kv_bytes_per_ns()
        self.assertGreater(r_hat, 0)
        epoch_before = self.tracker.quota_retry_key()[0]
        # link_id 5 ⇒ 端点 (2,3)（C++ 枚举序，2×4 mesh：dim0 正向第
        # 三条 = (2,3)；见 A5'/B3 换算测试的逐 id 断言）。
        self.scheduler._ingest_link_telemetry({
            "tick": 1000,
            "link_telemetry": [{
                "link_id": 5, "served_bytes": int(r_hat * 0.5 * 500),
                "active_ns": 500,
                "window_start_ns": 0, "window_end_ns": 1000}]})
        # 收缩：Q(3,2) 从 Q_init=2 减半至 1。
        self.assertEqual(self.tracker.link_quota((3, 2)), 1)
        self.assertGreater(
            self.tracker.quota_retry_key()[0], epoch_before)
        self.assertIn("shrink", self.scheduler._quota_aimd_action_counts)

    def test_no_enrolled_link_is_silent(self):
        """无在册流的链路不出现在喂入字典（无测量即无信号）。"""
        self.scheduler._ingest_link_telemetry({
            "tick": 1000,
            "link_telemetry": [{
                "link_id": 5, "served_bytes": 1000, "active_ns": 500,
                "window_start_ns": 0, "window_end_ns": 1000}]})
        self.assertEqual(self.tracker.link_quota((2, 3)), 2)
        self.assertEqual(self.scheduler._quota_aimd_action_counts, {})

    def test_static_mode_never_consumes_telemetry(self):
        """static 模式不喂 observe_telemetry（Q 恒 = Q_init）。"""
        scheduler = _scheduler(quota_mode="static")
        scheduler._ingest_link_telemetry({
            "tick": 1000,
            "link_telemetry": [{
                "link_id": 5, "served_bytes": 1, "active_ns": 500,
                "window_start_ns": 0, "window_end_ns": 1000}]})
        self.assertEqual(
            scheduler._quota_tracker.link_quota((2, 3)), 2)


# ============================ 5. port_snapshot 接通（步骤 6）==

class PortSnapshotTest(unittest.TestCase):
    @staticmethod
    def _audit_fields(scheduler):
        record = SelectionRecord(
            mode="joint", chosen=_candidate(0, "stay"),
            candidates=(_candidate(0, "stay", cost_ns=1),),
            instance_rule_note="note", remote_enabled=True)
        loads = {}
        for index in range(4):
            loads[index] = InstanceLoadView(
                instance_index=index,
                queued_task_load_ns=0, running_task_load_ns=0,
                active_decode_task_load_ns=0,
                hbm_remaining_bytes_by_tp_rank=(1, 1),
                reclaimable_bytes_by_tp_rank=(0, 0))
        cost_model = SimpleNamespace(loads=loads)
        return scheduler._joint_decision_audit_fields(
            record, cost_model, _session_view(), {"flows": {}})

    def test_quota_on_replaces_na_with_measured(self):
        scheduler = _scheduler()
        scheduler._hbm_ports.register(0, owner="t0")
        scheduler._hbm_ports.register(1, owner="t0")
        fields = self._audit_fields(scheduler)
        entries = fields["port_snapshot"]["instances"]
        self.assertEqual(len(entries), 4)
        first = entries[0]
        self.assertEqual(first["instance_index"], 0)
        self.assertEqual(first["u_port_active_decode_streams"], 0)
        self.assertEqual(
            first["u_port_registered_transfer_flows"], 1)
        self.assertEqual(first["u_port_total"], 1)
        self.assertEqual(first["bulk_slots_used"], 0)
        self.assertEqual(first["bulk_slots_cap"], 2)
        self.assertGreaterEqual(first["parity_gate_headroom"], 0)
        for entry in entries:
            for value in entry.values():
                self.assertNotEqual(value, "NA")

    def test_quota_off_keeps_frozen_na(self):
        scheduler = _scheduler(quota_mode="off")
        fields = self._audit_fields(scheduler)
        for entry in fields["port_snapshot"]["instances"]:
            for name in ("u_port_active_decode_streams",
                         "u_port_registered_transfer_flows",
                         "u_port_total", "bulk_slots_used",
                         "bulk_slots_cap", "parity_gate_headroom"):
                self.assertEqual(entry[name], "NA")


# ================================ 6. A5'/B3 rider：键换算 ==

class TelemetryEndpointKeyTest(unittest.TestCase):
    """LinkId → (src, dst) 换算：C++ MultiDimTopology 枚举配方逐 id
    断言（2×4 mesh ⇒ dims (4,2) ⇒ 20 条链路）。"""

    EXPECTED_PREFIX = {
        0: (0, 1), 1: (1, 0), 2: (1, 2), 3: (2, 1), 4: (2, 3), 5: (3, 2),
        6: (4, 5), 7: (5, 4), 8: (5, 6), 9: (6, 5), 10: (6, 7), 11: (7, 6),
        12: (0, 4), 13: (4, 0), 14: (1, 5), 15: (5, 1), 16: (2, 6),
        17: (6, 2), 18: (3, 7), 19: (7, 3),
    }

    def setUp(self) -> None:
        self.scheduler = _scheduler()

    def test_full_map_matches_cpp_enumeration(self):
        for link_id, endpoint in self.EXPECTED_PREFIX.items():
            self.assertEqual(
                self.scheduler._telemetry_endpoint_link_key(link_id),
                endpoint, link_id)
        with self.assertRaises(ValueError):
            self.scheduler._telemetry_endpoint_link_key(20)

    def test_ingest_keys_rates_by_endpoint(self):
        self.scheduler._ingest_link_telemetry({
            "tick": 100, "link_telemetry": [
                {"link_id": 5, "served_bytes": 1000, "active_ns": 500,
                 "window_start_ns": 0, "window_end_ns": 100},
                {"link_id": 14, "served_bytes": 500, "active_ns": 500,
                 "window_start_ns": 0, "window_end_ns": 100}]})
        self.assertEqual(
            self.scheduler._link_telemetry_rates,
            {(3, 2): 2.0, (1, 5): 1.0})

    def test_stub_identity_map_passthrough(self):
        """F6 销账：hasattr 透传软门已删——整型键形态改由替身显式预置
        恒等映射 {i: i}（等价旧 C7 期退化路径，语义自担；属性缺席现在
        = AttributeError fail-loud，不再静默透传）。"""
        self.scheduler._telemetry_link_id_map = {5: 5}
        self.assertEqual(
            self.scheduler._telemetry_endpoint_link_key(5), 5)
        self.scheduler._ingest_link_telemetry({
            "tick": 100, "link_telemetry": [
                {"link_id": 5, "served_bytes": 1000, "active_ns": 500,
                 "window_start_ns": 0, "window_end_ns": 100}]})
        self.assertEqual(self.scheduler._link_telemetry_rates, {5: 2.0})

    def test_stub_missing_map_attribute_fails_loud(self):
        """F6 销账钉死：替身漏设 _telemetry_link_id_map（且未删净的旧
        写法 del）必须 AttributeError——映射装配不可静默缺席；断言
        锚定缺失属性名（别的属性先炸不许冒名通过）。"""
        del self.scheduler._telemetry_link_id_map
        with self.assertRaisesRegex(
                AttributeError, "_telemetry_link_id_map"):
            self.scheduler._telemetry_endpoint_link_key(5)


# ==================================== 7. F7 耦合规则（发射层）==

class F7CouplingRuleTest(unittest.TestCase):
    """aimd ⇒ --link-telemetry 自动注入（runner 半）+ 防御性断言
    （.sh 半）。"""

    def test_runner_injects_link_telemetry_for_aimd(self):
        """F4（复审修复）：原 skipUnless 二进制存在 ⇒ 本仓（无 build）
        恒 skip、注入断言零效。改为 tmp 占位可执行文件 + monkeypatch
        joint_runner 的二进制路径常量（_REPO_ROOT/_BINARY_RELPATH 查找）
        指向它——内层 subprocess.run 已被假函数替换，二进制从不真跑，
        占位只需通过 exists/X_OK/sha256 三道启动前检查。"""
        import importlib
        from pathlib import Path
        runner = importlib.import_module("joint_runner")
        saved_env = dict(os.environ)
        captured = {}
        run_dir = tempfile.mkdtemp(prefix="c11_runner_")
        # 模块级常量一并转存：锁/runs 目录同样指入 tmp（不在仓内落
        # .single_simulation.lock）。
        saved_module_state = {
            name: getattr(runner, name)
            for name in ("_REPO_ROOT", "_RUNS_DIR", "_LOCK_PATH")}

        def _fake_inner_run(argv, **kwargs):
            captured["env"] = dict(os.environ)
            return SimpleNamespace(returncode=0)

        original_run = subprocess.run
        try:
            subprocess.run = _fake_inner_run
            stub_root = Path(run_dir) / "stub_repo"
            stub_binary = stub_root.joinpath(*runner._BINARY_RELPATH.split("/"))
            stub_binary.parent.mkdir(parents=True, exist_ok=True)
            stub_binary.write_bytes(b"placeholder (never executed)\n")
            stub_binary.chmod(0o755)
            runner._REPO_ROOT = stub_root
            runner._RUNS_DIR = Path(run_dir) / "stub_runs"
            runner._LOCK_PATH = runner._RUNS_DIR / ".single_simulation.lock"
            csv_path = os.path.join(run_dir, "queue.csv")
            with open(csv_path, "w", encoding="utf-8") as sink:
                sink.write("placeholder\n")
            rc = runner.main([
                os.path.join(run_dir, "run"), csv_path, "--quota", "aimd"])
            self.assertEqual(rc, 0)
            # 注入事实：SH_LINK_TELEMETRY=1 已置位（内层发射路径可见）
            # + invocation.json 记录。
            self.assertEqual(captured["env"].get("SH_LINK_TELEMETRY"), "1")
            self.assertEqual(
                captured["env"].get("JOINT_QUOTA_MODE"), "aimd")
            with open(os.path.join(run_dir, "run", "invocation.json"),
                      encoding="utf-8") as source:
                invocation = json.load(source)
            self.assertTrue(invocation["link_telemetry_injected"])
            self.assertEqual(
                invocation["joint_switches"]["JOINT_QUOTA_MODE"], "aimd")
            # A12' 批（G4）：占位二进制 sha 锚定——invocation.json 记录
            # 的 binary_sha256 必须等于占位内容 sha256（runner 半的绑定
            # sanity；C++ 接口漂移的真防护 = F5 真 commit 路径验证 +
            # 13 臂矩阵真跑，本测试不冒充）。
            import hashlib
            with open(stub_binary, "rb") as handle:
                self.assertEqual(
                    invocation["binary_sha256"],
                    hashlib.sha256(handle.read()).hexdigest())
        finally:
            # F4（复审修复）：全局猴子补丁恢复移入 finally——上面任何
            # 断言/runner.main 抛异常时不再泄漏 subprocess.run 假函数。
            subprocess.run = original_run
            for name, value in saved_module_state.items():
                setattr(runner, name, value)
            for name in list(os.environ):
                if name not in saved_env:
                    del os.environ[name]
            os.environ.update(saved_env)
            shutil.rmtree(run_dir, ignore_errors=True)

    def test_shell_defensive_assertion(self):
        """run_online_strategy.sh：aimd + 无遥测 ⇒ fail-closed（注入
        逻辑损坏才触发）；非 aimd 无遥测 = 既有路径（不因本断言早退）。
        提取注入段在隔离 bash 中执行（脚本其余段有 GEN_MATCH 副作用）。"""
        script = os.path.join(
            os.path.abspath(_RUN_SCRIPTS_DIR), "run_online_strategy.sh")
        with open(script, encoding="utf-8") as source:
            text = source.read()
        start = text.index("LINK_TELEMETRY_ARGS=()")
        exit_idx = text.index("exit 1", start)
        end = text.index("fi", exit_idx) + 2
        block = text[start:end]
        for quota, telemetry, expect_fail in (
                ("aimd", None, True),
                ("aimd", "1", False),
                ("static", None, False),
                (None, "1", False)):
            env_line = "JOINT_QUOTA_MODE={}".format(quota)
            if quota is None:
                env_line = "unset JOINT_QUOTA_MODE"
            telemetry_line = (
                "SH_LINK_TELEMETRY={}".format(telemetry)
                if telemetry is not None else "unset SH_LINK_TELEMETRY")
            proc = subprocess.run(
                ["bash", "-c", "{}; {}; {}".format(
                    env_line, telemetry_line, block)],
                capture_output=True, text=True)
            if expect_fail:
                self.assertNotEqual(proc.returncode, 0)
                self.assertIn("requires link telemetry", proc.stderr)
            else:
                self.assertEqual(proc.returncode, 0, proc.stderr)

    def test_stress_fixture_gate_fails_on_export_sentinels(self):
        """joint_capacity_stress_fixture.sh（A13'/H1，2026-09-22 第三轮
        复审）：G3 逐键哨兵（<key>_export_error）在场 = 对应键导出失败
        = 证据链断裂——硬门禁必须 FAIL（旧行为：deep=None 使 "not deep"
        恒真 ⇒ 假 GREEN）。文本级接线钉：执行级验证需完整 run 目录夹
        具（决策日志/侧车/SLO 三工件），成本与收益不成比例；此处钉
        哨兵扫描与门禁项的接线契约 + 键名拼接与生产端一致。"""
        script = os.path.join(
            os.path.abspath(_RUN_SCRIPTS_DIR),
            "joint_capacity_stress_fixture.sh")
        with open(script, encoding="utf-8") as source:
            text = source.read()
        # 哨兵扫描在场：侧车顶层 <key>_export_error 键收集。
        self.assertIn('key.endswith("_export_error")', text)
        # 硬门禁表达式含"无哨兵"项（与侧车在场/deep_gap 空同列）。
        start = text.index("sys.exit(0 if")
        end = text.index("else 2)", start)
        gate = text[start:end]
        self.assertIn("not export_errors", gate)
        self.assertIn("sidecar_present", gate)
        self.assertIn("not deep", gate)
        # 键名拼接规则与生产端 dump_joint_kv_ledgers 一致
        # （payload[key + "_export_error"]，四键之任一）。
        online_dir = os.path.join(
            os.path.dirname(os.path.abspath(_RUN_SCRIPTS_DIR)),
            "workload", "llama2_7b_inference", "online")
        with open(os.path.join(online_dir, "online_service.py"),
                  encoding="utf-8") as source:
            producer = source.read()
        self.assertIn('"_export_error"', producer)

    def test_shell_appends_flag_when_telemetry_set(self):
        """SH_LINK_TELEMETRY=1 ⇒ 注入段构造的 --link-telemetry 真占
        独立参数位。

        F4（复审修复）：原纯文本 grep（只查脚本源含两子串）零效——改
        提取执行（同 test_shell_defensive_assertion 范式）：注入段后拼
        NUL 分隔的数组展开探针，核对旗标逐参数位出现/缺席。"""
        script = os.path.join(
            os.path.abspath(_RUN_SCRIPTS_DIR), "run_online_strategy.sh")
        with open(script, encoding="utf-8") as source:
            text = source.read()
        start = text.index("LINK_TELEMETRY_ARGS=()")
        exit_idx = text.index("exit 1", start)
        end = text.index("fi", exit_idx) + 2
        block = text[start:end]
        # 参数位探针：模拟启动行 "${LINK_TELEMETRY_ARGS[@]}" 的实参
        # 展开（每参数一个 NUL 终止符——参数边界可观测）。
        probe = 'printf "%s\\0" "${LINK_TELEMETRY_ARGS[@]}"'
        # (quota, telemetry, expected_argv)
        # aimd+1：runner 注入后的标准形态 → 恰一个旗标参数。
        # 未置 quota+1：直跑遥测臂（非 aimd 路径）→ 同样恰一个。
        # static+未置遥测：合法无遥测路径 → 空参数向量（无旗标、无
        #   空串参数）。
        for quota, telemetry, expected_argv in (
                ("aimd", "1", ["--link-telemetry"]),
                (None, "1", ["--link-telemetry"]),
                ("static", None, [])):
            env_line = ("JOINT_QUOTA_MODE={}".format(quota)
                        if quota is not None else "unset JOINT_QUOTA_MODE")
            telemetry_line = (
                "SH_LINK_TELEMETRY={}".format(telemetry)
                if telemetry is not None else "unset SH_LINK_TELEMETRY")
            proc = subprocess.run(
                ["bash", "-c", "{}; {}; {}; {}".format(
                    env_line, telemetry_line, block, probe)],
                capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            argv = [part for part in proc.stdout.split("\0") if part != ""]
            self.assertEqual(argv, expected_argv, (quota, telemetry))


if __name__ == "__main__":
    unittest.main()
