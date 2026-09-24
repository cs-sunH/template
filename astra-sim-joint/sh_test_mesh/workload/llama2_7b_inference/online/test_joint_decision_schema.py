#!/usr/bin/env python3
"""test_joint_decision_schema.py -- C5（WP6a，F8 冻结）决策日志 schema 钉子。

背景（2026-09-22，joint 改造 C5）：joint_admission 决策行扩展为可重建
"每决策 × 每候选 × 每字段"的完整决策时刻审计流。schema 先行冻结
（F8）、先于 WP3 落地；配额类字段（port_snapshot 族）本卡记 "NA"
（C2/C11 接通后替换，不缺席）。breakdown 值由 C2 物化填充——本卡判据
= 字段存在性与可解析性（值可为 None）。

钉子：
  1. 全路径（真实 _try_admit_request，turn-0）：决策行含全部审计扩展
     键，逐候选 hop/breakdown 可重建，整行 JSON 可序列化/解析重建；
  2. 有历史场景（真实 cost model + 真实 select_instance_and_action，
     LOCAL 基驻留实例 1）：适用动作集四动作齐、copy/remote 候选 breakdown
     11 字段在场、hop 语义（驻留实例 0 跳/跨实例 >0、与 breakdown.hops
     冗余一致）；
  3. schema 冻结锚：SH 冻结字段序与 JCM ActionCostBreakdown 声明逐一
     同名（防字段漂移）；
  4. selected_action 枚举 + recompute elected/forced 分位（合成记录覆
     盖两 forced 原因；现行三态模型下 forced 不可达——REMOTE 基可经池
     恢复，schema 先行冻结枚举）；
  5. 计数语义（设计文档 §5.2）：elected/forced 两口径分列可导出、
     forced 不被静默丢弃出分母、不得混报；
  6. 流表快照摘要 = 每链路登记流数（背景流在决策行可见）；
     port_snapshot 六字段逐实例 "NA" 不缺席。

披露边界（冻结）：本 schema 只含决策时刻记录；结算时刻的逐请求合并
披露走 FS 侧 kv_delta_journal（C14 步骤 5），不在本测试范围。

运行：cd sh_test_mesh/workload/llama2_7b_inference &&
      python3 online/test_joint_decision_schema.py   （或 pytest 同路径）
"""
import json
import os
import sys
import unittest
from types import SimpleNamespace

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
from joint.joint_cost_model import (  # noqa: E402
    ACTION_ORDER,
    ActionCandidate,
    ActionCostBreakdown,
    CausalHorizonEstimator,
    JointHardwareRates,
    LinkFlowRegistry,
    RequestView,
    ServiceFactors,
    SessionKVView,
)
from joint.joint_scheduler import select_instance_and_action  # noqa: E402
from joint.joint_config import parse_joint_config  # noqa: E402
from joint.hbm_port_flow_registry import (  # noqa: E402  F6 销账替身对齐
    HbmPortFlowRegistry,
)
from online.sh30_online_scheduler import (  # noqa: E402
    _JOINT_BREAKDOWN_LOG_FIELDS,
    _OnlineInstanceState,
    _OnlineRequestRuntime,
    _PoolPortRegistry,
    _TASK_LOAD_CACHE_CAPACITY,  # F6 销账：替身对齐 __init__ 初值
    Sh30OnlineScheduler,
)

#: F8 冻结的 selected_action 枚举（四动作，不设第五动作——F10）。
SELECTED_ACTION_ENUM = ("stay", "recompute", "copy", "remote-read")
#: port_snapshot 的六个配额类字段（WP3 前恒 "NA"）。
PORT_SNAPSHOT_NA_FIELDS = (
    "u_port_active_decode_streams", "u_port_registered_transfer_flows",
    "u_port_total", "bulk_slots_used", "bulk_slots_cap",
    "parity_gate_headroom",
)
#: load_view 的五个负载量字段（InstanceLoadView 构造点口径）。
LOAD_VIEW_FIELDS = (
    "queued_task_load_ns", "running_task_load_ns",
    "active_decode_task_load_ns", "hbm_remaining_bytes_by_tp_rank",
    "reclaimable_bytes_by_tp_rank",
)


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


def _make_scheduler() -> Sh30OnlineScheduler:
    """绕过 __init__（需 manifest/graph/bridge），装配 _try_admit_request
    全路径（成功段）用到的属性（Roofline 小参数与
    test_joint_decode_context.py / test_admit_gate.py 同款）。"""
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
        _OnlineInstanceState(index=i)
        for i in range(len(scheduler.topology.instances))
    ]
    scheduler.hardware = hardware
    scheduler.model = model
    scheduler.kv_manager = KVCacheManager(scheduler.topology, model)
    scheduler.p_chunk = 512
    scheduler._joint_horizon = CausalHorizonEstimator(
        cold_start_default_tokens=1)
    scheduler._joint_flows = LinkFlowRegistry()
    scheduler._pool_ports = _PoolPortRegistry()
    # 对齐 __init__ 初值（F6 销账：软门已删，替身漏设 = AttributeError；
    # _register/_release_transfer_flows 直达 _hbm_ports）。
    scheduler._hbm_ports = HbmPortFlowRegistry()
    scheduler._joint_factors = ServiceFactors()
    scheduler._joint_rates = JointHardwareRates.from_gbps(
        noc_link_gbps=hardware.d2d_bandwidth_gbps,
        pool_port_gbps=50.0,
        local_hbm_gbps=hardware.local_hbm_bandwidth_gbps,
        d2d_latency_ns=int(hardware.d2d_latency_ns),
        pool_latency_ns=0,
    )
    scheduler._instance_edge_ports = {}
    scheduler.joint_config = parse_joint_config(env={})
    scheduler._joint_mode = scheduler.joint_config.scheduler_mode
    scheduler._prefill_task_cache = {}
    scheduler._decode_task_load_cache = {}
    # 对齐 __init__ 初值（F6 销账：软门已删，替身漏设 = AttributeError）。
    scheduler._task_load_cache_capacity = _TASK_LOAD_CACHE_CAPACITY
    scheduler._quota_tracker = None  # off 档（parse_joint_config 缺省）
    scheduler._link_telemetry_rates = {}
    # M1：C8 遥测状态 __init__ 同款（时间加权活跃流数）。
    scheduler._link_telemetry_flow_counts = {}  # C8 遥测状态 __init__ 同款
    scheduler._telemetry_last_tick_ns = 0
    # C11 决策时延埋点账本（对齐 __init__ 初值，F6 销账：软门已删，
    # _account_decision_wall 直加——替身漏设 = AttributeError）。
    scheduler._admission_decision_wall_ns_total = 0
    scheduler._admission_decision_wall_ns_max = 0
    scheduler._admission_decision_count = 0
    scheduler._quota_verdict_wall_ns_total = 0
    scheduler._quota_verdict_candidate_checks = 0
    # C11 动作选中计数（对齐 __init__ 初值，F6 销账：_note_action_
    # selection 直加计数——替身漏设 = AttributeError）。
    scheduler._joint_action_selection_counts = {
        "stay": 0, "copy": 0, "remote-read": 0,
        "recompute_elected": 0,
        "recompute_forced_no_history": 0,
        "recompute_forced_quota_deferred": 0,
        "recompute_forced_evicted_permanent": 0,
    }
    scheduler._snapshot_verify = False
    scheduler._kv_ledger_epoch = 0
    scheduler._ready_frontier = set()
    # 决策日志（基类 log_decision：无 sink 时缓冲 online_log_rows）。
    scheduler.online_log_rows = []
    scheduler.online_log_count = 0
    scheduler.decision_log_sink = None
    # 发射/账本尾段（_emit_admission / _ledger_* 的最小账面）。
    scheduler.graph = _GraphStub()
    scheduler.ledger_admitted = {}
    scheduler.ledger_issued = {}
    scheduler._emitted_by_delivery = {0: {"requests": []}}
    scheduler._batch = {
        "assignments": [], "watches": [], "delivery_sequence": 0}
    return scheduler


def _runtime(request_id="r_schema", session_id="s_schema",
             turn_index=0, input_tokens=64):
    runtime = _OnlineRequestRuntime({
        "request_id": request_id,
        "session_id": session_id,
        "turn_index": turn_index,
        "queue_index": 0,
        "prefill_length": input_tokens,
        "decode_length": 4,
        "history_tokens_before": 0 if turn_index == 0 else 512,
        "prefill_context_tokens": (
            input_tokens if turn_index == 0 else 512 + input_tokens),
        "final_context_tokens": (
            input_tokens + 4 if turn_index == 0 else 512 + input_tokens + 4),
    }, 512)
    runtime.estimated_arrival_ns = 0
    return runtime


def _history_session_view() -> SessionKVView:
    """LOCAL 基驻留实例 1 的决策时点会话视图（1024 token 全层驻留）。"""
    shards = (2048, 2048)
    return SessionKVView(
        session_id="s_hist", home_instance=1, resident_instance=1,
        location="local_hbm", history_tokens=1024,
        resident_prefix_layers=2,
        history_bytes_by_tp_rank=shards,
        missing_bytes_by_tp_rank=(0, 0))


def _history_request_view() -> RequestView:
    return RequestView(
        request_id="r_hist", session_id="s_hist", input_tokens=64,
        history_tokens_before=1024, estimated_decode_tokens=4,
        horizon_source="cold_start_default",
        input_kv_bytes_by_tp_rank=(128, 128))


def _admission_rows(scheduler):
    return [row for row in scheduler.online_log_rows
            if row["kind"] == "joint_admission"]


class JointDecisionSchemaFullPathTest(unittest.TestCase):
    """C5：真实 _try_admit_request 全路径的决策行 schema。"""

    def setUp(self) -> None:
        self.scheduler = _make_scheduler()

    def test_turn0_row_reconstructible_per_decision_candidate_field(self):
        """turn-0 准入：审计扩展键全在场，逐候选 hop/breakdown 可重建，
        整行 JSON 可序列化/解析重建（每决策 × 每候选 × 每字段）。"""
        runtime = _runtime()
        self.assertTrue(self.scheduler._try_admit_request(runtime, 0))
        rows = _admission_rows(self.scheduler)
        self.assertEqual(len(rows), 1)
        decision = rows[0]["decision"]
        for key in ("selected_action", "applicable_actions",
                    "recompute_selection", "load_view", "flow_snapshot",
                    "port_snapshot"):
            self.assertIn(key, decision, f"missing audit key {key}")
        # 既有字段共存不冲突（C3/C3b 交付保持）。
        for key in ("joint_mode", "joint_action", "joint_instance_index",
                    "joint_cost_ns", "candidates", "contention_coverage"):
            self.assertIn(key, decision)
        candidates = decision["candidates"]
        self.assertEqual(len(candidates), 2 * len(ACTION_ORDER))
        for candidate in candidates:
            self.assertIn("hops", candidate)
            self.assertIsInstance(candidate["hops"], int)
            self.assertIn("breakdown", candidate)
            if candidate["applicable"]:
                # 字段存在性与可解析性（值可为 None——物化在 C2）。
                self.assertIsInstance(candidate["breakdown"], dict)
                self.assertEqual(
                    sorted(candidate["breakdown"]),
                    sorted(_JOINT_BREAKDOWN_LOG_FIELDS))
                for name in _JOINT_BREAKDOWN_LOG_FIELDS:
                    value = candidate["breakdown"][name]
                    if name == "notes":
                        self.assertIsInstance(value, list)
                    else:
                        self.assertTrue(value is None
                                        or isinstance(value, int))
                # 冗余断言位：候选级 hops 与 breakdown.hops 同值。
                self.assertEqual(
                    candidate["hops"], candidate["breakdown"]["hops"])
            else:
                self.assertIsNone(candidate["breakdown"])
                self.assertIsInstance(
                    candidate["inapplicable_reason"], str)
        # 整行 JSON round-trip（可解析重建判据）。
        reparsed = json.loads(json.dumps(rows[0]))
        self.assertEqual(reparsed, rows[0])

    def test_turn0_selected_action_and_applicable_set(self):
        """turn-0：selected_action ∈ 枚举且与 joint_action 同值（显式冗余
        断言位）；适用动作集 = {stay, recompute}（无历史 → copy 不适用、
        无异地驻留 → remote 不适用）；非 recompute 决策分位记 None。"""
        runtime = _runtime()
        self.assertTrue(self.scheduler._try_admit_request(runtime, 0))
        decision = _admission_rows(self.scheduler)[0]["decision"]
        self.assertIn(decision["selected_action"], SELECTED_ACTION_ENUM)
        self.assertEqual(decision["selected_action"],
                         decision["joint_action"])
        self.assertEqual(decision["selected_action"], "stay")
        self.assertEqual(decision["applicable_actions"],
                         ["stay", "recompute"])
        # 适用动作集可由 candidates 导出（一致性）。
        derived = [
            action for action in ACTION_ORDER
            if any(c["action"] == action and c["applicable"]
                   for c in decision["candidates"])]
        self.assertEqual(decision["applicable_actions"], derived)
        self.assertIsNone(decision["recompute_selection"])

    def test_turn0_load_view_flow_snapshot_port_snapshot(self):
        """负载视图（instance_index + 五个负载量字段，逐实例）、流表
        快照摘要（每链路登记流数——背景流可见）、端口快照（六字段逐
        实例 NA 不缺席）。"""
        # 背景流：另一 owner 在链路 0->1 登记 2 流。
        self.scheduler._joint_flows.register_path((0, 1), owner="bg#1")
        self.scheduler._joint_flows.register_path((0, 1), owner="bg#2")
        runtime = _runtime()
        self.assertTrue(self.scheduler._try_admit_request(runtime, 0))
        decision = _admission_rows(self.scheduler)[0]["decision"]
        # 负载视图：两实例 × (instance_index + 五字段)。
        load_view = decision["load_view"]
        self.assertEqual(len(load_view), 2)
        for entry in load_view:
            self.assertEqual(
                sorted(entry),
                sorted(("instance_index",) + LOAD_VIEW_FIELDS))
            self.assertIsInstance(entry["hbm_remaining_bytes_by_tp_rank"],
                                  list)
            self.assertEqual(
                len(entry["hbm_remaining_bytes_by_tp_rank"]), 2)
            self.assertEqual(
                len(entry["reclaimable_bytes_by_tp_rank"]), 2)
        # 流表快照：每链路登记流数（"src->dst" 键）。
        self.assertEqual(decision["flow_snapshot"], {"0->1": 2})
        # 端口快照：逐实例六字段全 NA（WP3 前占位，不缺席）。
        port_snapshot = decision["port_snapshot"]
        self.assertEqual(len(port_snapshot["instances"]), 2)
        for entry in port_snapshot["instances"]:
            self.assertEqual(
                sorted(entry),
                sorted(("instance_index",) + PORT_SNAPSHOT_NA_FIELDS))
            for name in PORT_SNAPSHOT_NA_FIELDS:
                self.assertEqual(entry[name], "NA")


class JointDecisionSchemaHistoryTest(unittest.TestCase):
    """C5：有历史场景（真实 cost model + 真实选择器）的审计字段。"""

    def setUp(self) -> None:
        self.scheduler = _make_scheduler()

    def _real_record(self):
        session_view = _history_session_view()
        request_view = _history_request_view()
        cost_model = self.scheduler._joint_cost_model(
            0, decode_context_tokens=1024 + 64, decode_average_length=1)
        record = select_instance_and_action(
            mode=self.scheduler._joint_mode,
            cost_model=cost_model,
            session=session_view,
            request=request_view,
            remote_enabled=True)
        return record, cost_model, session_view

    def test_history_applicable_set_and_breakdowns(self):
        """LOCAL 基驻留实例 1：适用动作集四动作齐；异地实例的 copy/
        remote-read 候选 breakdown 11 字段在场（字段存在性判据）。"""
        record, cost_model, session_view = self._real_record()
        audit = self.scheduler._joint_decision_audit_fields(
            record, cost_model, session_view, cost_model.flow_registry
            .snapshot())
        self.assertEqual(audit["applicable_actions"], list(ACTION_ORDER))
        self.assertIn(audit["selected_action"], SELECTED_ACTION_ENUM)
        # 逐候选 hop + breakdown（applicable 全 11 字段）。
        for candidate in record.candidates:
            hops = self.scheduler._joint_decision_route_hops(
                cost_model, session_view, candidate.instance_index)
            self.assertIsInstance(hops, int)
            if candidate.instance_index == session_view.resident_instance:
                self.assertEqual(hops, 0)
            else:
                self.assertGreater(hops, 0)
            if candidate.applicable:
                breakdown = self.scheduler._joint_breakdown_log_dict(
                    candidate.breakdown)
                self.assertEqual(
                    sorted(breakdown), sorted(_JOINT_BREAKDOWN_LOG_FIELDS))
                # C1 后真实值在场（本卡只判存在性/可解析性，不判数值）。
                self.assertEqual(breakdown["hops"], hops)
        # JSON 可解析性（审计字段块 round-trip）。
        self.assertEqual(json.loads(json.dumps(audit)), audit)

    def test_load_view_matches_cost_model_inputs(self):
        """负载视图 = InstanceLoadView 构造点数据（决策输入同刻快照）。"""
        record, cost_model, session_view = self._real_record()
        audit = self.scheduler._joint_decision_audit_fields(
            record, cost_model, session_view, {})
        by_index = {entry["instance_index"]: entry
                    for entry in audit["load_view"]}
        self.assertEqual(sorted(by_index), [0, 1])
        for index, view in cost_model.loads.items():
            entry = by_index[index]
            self.assertEqual(entry["queued_task_load_ns"],
                             view.queued_task_load_ns)
            self.assertEqual(entry["running_task_load_ns"],
                             view.running_task_load_ns)
            self.assertEqual(entry["active_decode_task_load_ns"],
                             view.active_decode_task_load_ns)
            self.assertEqual(
                entry["hbm_remaining_bytes_by_tp_rank"],
                list(view.hbm_remaining_bytes_by_tp_rank))
            self.assertEqual(
                entry["reclaimable_bytes_by_tp_rank"],
                list(view.reclaimable_bytes_by_tp_rank))


class JointDecisionSchemaFreezeTest(unittest.TestCase):
    """C5：schema 冻结锚与 selected_action/分位语义。"""

    def setUp(self) -> None:
        self.scheduler = _make_scheduler()

    def test_frozen_field_list_matches_jcm_breakdown_declaration(self):
        """SH 冻结字段序与 JCM ActionCostBreakdown 声明逐一同名（防
        字段漂移；全 11 字段）。"""
        declared = tuple(ActionCostBreakdown.__dataclass_fields__)
        self.assertEqual(len(_JOINT_BREAKDOWN_LOG_FIELDS), 11)
        self.assertEqual(sorted(_JOINT_BREAKDOWN_LOG_FIELDS),
                         sorted(declared))

    def _record_with_chosen(self, *, chosen_action, applicable_actions):
        """合成 SelectionRecord：控制适用动作集与选中动作（分位语义
        单元覆盖——现行在线路径 forced 不可达，schema 枚举先冻结）。"""
        candidates = []
        for instance_index in (0, 1):
            for action in ACTION_ORDER:
                applicable = action in applicable_actions
                candidates.append(ActionCandidate(
                    instance_index=instance_index, action=action,
                    applicable=applicable,
                    inapplicable_reason=(
                        None if applicable else "synthetic_inapplicable"),
                    cost_ns=(42 if applicable else None),
                    breakdown=(ActionCostBreakdown(
                        target_wait_ns=1, history_prep_ns=2,
                        eviction_wait_ns=3, compute_ns=4,
                        remote_read_ns=5, merge_ns=6,
                        contention_divisor=1, hops=0, notes=("synthetic",),
                        remote_read_first_credit_ns=0,
                        remote_read_stream_ns=0)
                        if applicable else None)))
        chosen = next(candidate for candidate in candidates
                      if candidate.action == chosen_action
                      and candidate.instance_index == 0)
        return SimpleNamespace(chosen=chosen, candidates=tuple(candidates))

    def _audit(self, record, history_tokens):
        session_view = SimpleNamespace(history_tokens=history_tokens)
        cost_model = SimpleNamespace(
            loads={0: SimpleNamespace(
                instance_index=0, queued_task_load_ns=0,
                running_task_load_ns=0, active_decode_task_load_ns=0,
                hbm_remaining_bytes_by_tp_rank=(0,),
                reclaimable_bytes_by_tp_rank=(0,))})
        return self.scheduler._joint_decision_audit_fields(
            record, cost_model, session_view, {})

    def test_elected_when_other_applicable_action_exists(self):
        """真选中：至少一个其余适用动作存在、联合比较仍选中 recompute
        ——tier=elected、forced_reason=None。"""
        record = self._record_with_chosen(
            chosen_action="recompute",
            applicable_actions={"stay", "recompute"})
        audit = self._audit(record, history_tokens=512)
        self.assertEqual(audit["selected_action"], "recompute")
        self.assertEqual(audit["recompute_selection"],
                         {"tier": "elected", "forced_reason": None})

    def test_forced_reason_no_history(self):
        """必经重算（首轮，无任何历史 KV）：适用集={recompute} 且
        history=0 → tier=forced / reason=no_history。"""
        record = self._record_with_chosen(
            chosen_action="recompute",
            applicable_actions={"recompute"})
        audit = self._audit(record, history_tokens=0)
        self.assertEqual(audit["recompute_selection"],
                         {"tier": "forced", "forced_reason": "no_history"})

    def test_forced_reason_evicted_permanent(self):
        """必经重算（历史已被永久驱逐、无有效后备副本）：适用集=
        {recompute} 且 history>0 → tier=forced / reason=
        evicted_permanent（现行三态模型该态不可达，schema 先冻结枚举）。"""
        record = self._record_with_chosen(
            chosen_action="recompute",
            applicable_actions={"recompute"})
        audit = self._audit(record, history_tokens=512)
        self.assertEqual(
            audit["recompute_selection"],
            {"tier": "forced", "forced_reason": "evicted_permanent"})

    def test_non_recompute_selection_records_null_tier(self):
        """非 recompute 决策：分位字段记 None（枚举位不缺席、不误报）。"""
        record = self._record_with_chosen(
            chosen_action="copy",
            applicable_actions={"stay", "recompute", "copy"})
        audit = self._audit(record, history_tokens=512)
        self.assertEqual(audit["selected_action"], "copy")
        self.assertIsNone(audit["recompute_selection"])

    def test_counting_semantics_elected_and_forced_never_mixed(self):
        """计数语义（设计文档 §5.2）：从决策行流可导出 elected/forced
        两口径分列计数——recompute 选中数仅含 elected，forced 单列（按
        原因分计）、不并入选中数、不被静默丢弃出分母。"""
        rows = []
        for record, history in (
            (self._record_with_chosen(
                chosen_action="recompute",
                applicable_actions={"stay", "recompute"}), 512),
            (self._record_with_chosen(
                chosen_action="recompute",
                applicable_actions={"recompute"}), 0),
            (self._record_with_chosen(
                chosen_action="recompute",
                applicable_actions={"recompute"}), 512),
            (self._record_with_chosen(
                chosen_action="recompute",
                applicable_actions={"stay", "recompute", "copy"}), 512),
        ):
            rows.append(self._audit(record, history))
        # 全部为 recompute 决策行。
        self.assertTrue(all(row["selected_action"] == "recompute"
                            for row in rows))
        tiers = [row["recompute_selection"]["tier"] for row in rows]
        self.assertEqual(tiers, ["elected", "forced", "forced", "elected"])
        # recompute 选中数（elected 口径）= 2——不含 forced。
        elected = sum(1 for row in rows
                      if row["recompute_selection"]["tier"] == "elected")
        self.assertEqual(elected, 2)
        # forced 单列计数披露（按原因分计）、不并入 elected。
        forced_rows = [row for row in rows
                       if row["recompute_selection"]["tier"] == "forced"]
        self.assertEqual(len(forced_rows), 2)
        by_reason = {}
        for row in forced_rows:
            reason = row["recompute_selection"]["forced_reason"]
            self.assertIn(reason, (
                "no_history", "quota_deferred", "evicted_permanent"))
            by_reason[reason] = by_reason.get(reason, 0) + 1
        self.assertEqual(by_reason,
                         {"no_history": 1, "evicted_permanent": 1})
        # 分母语义：forced 不被静默丢弃——全部 recompute 决策行可数。
        self.assertEqual(elected + len(forced_rows), len(rows))
        # 两口径不得混报：elected ≠ 全体 recompute 行数（forced 在场）。
        self.assertNotEqual(elected, len(rows))


if __name__ == "__main__":
    unittest.main(verbosity=2)
