#!/usr/bin/env python3
"""test_growth_reservation_hold.py -- 先在潜伏缺陷修复的钉测（主 agent
裁决 2026-10-06 修法 a）。

被测语义：growth-only 多源远读（D remote-read@第三目标）在
``_on_prefill_drain`` 的预约注销点不再触发
"released HBM reservation before final KV allocation"——镜像
``joint_rebuild_reservation_hold`` 的机制：

  * drain 点：阶段预约被 hold（目标侧后缀未物化前不得释放），decode
    lease 正常承接（读流暂存有持有者）；
  * 首个目标侧列车核销点：物化恢复后缀基数（shard truth = 物理基
    后缀；本列车 decode 增长仍走 _joint_grow_decode 自身容量路径），
    随后注销预约（extra 收敛 0），hold 清位；
  * 足迹口径半边：D 阶段多源远读的后缀分量按物理基派生（U 后缀留
    i_P 经读流），不随暂存 tile 裁决的联合区间 (H+U) 膨胀。

负例：物化前直接释放仍被守卫拒绝（不掩盖真实泄漏）。

运行：cd sh_test_mesh/workload/llama2_7b_inference &&
      PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -p no:cacheprovider \\
          online/test_growth_reservation_hold.py -q
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
    FaceHardware,
    FaceInstanceSpec,
    FaceModel,
    KVCacheManager,
    build_instances,
)
from joint.hbm_port_flow_registry import HbmPortFlowRegistry  # noqa: E402
from joint.joint_config import parse_joint_config  # noqa: E402
from joint.joint_cost_model import (  # noqa: E402
    ActionCandidate,
    CausalHorizonEstimator,
    JointHardwareRates,
    LinkFlowRegistry,
    ServiceFactors,
)
from joint.joint_scheduler import SelectionRecord  # noqa: E402
from joint.predictor_cost_accounting import (  # noqa: E402
    PredictorCostAccounting,
)
from online.graph_batch_builder import GraphBatchBuilder  # noqa: E402
from online.online_scheduler_base import (  # noqa: E402
    _PROPAGATING_TAIL_SOURCE_ARRIVALS,
    _PROPAGATING_TAIL_SOURCE_DELIVERY,
    _PROPAGATING_TAIL_SOURCE_KV,
    OnlineSchedulerBase,
    PropagatingTailTracker,
)
from online.sh30_online_scheduler import (  # noqa: E402
    _OnlineInstanceState,
    _PoolPortRegistry,
    _TASK_LOAD_CACHE_CAPACITY,
    Sh30OnlineScheduler,
)
import online.sh30_online_scheduler as sh30  # noqa: E402
from types import SimpleNamespace  # noqa: E402

CTX = 128
INPUT_TOKENS = 50
DECODE = 16
RID = "r1"
SID = "s"


def _drive():
    """三实例 growth-only@第三目标的完整真实序列（提交环 + ack 簿记）。"""
    model = FaceModel(
        layers=16, hidden_size=4, ffn_size=4, num_heads=2, vocab_size=4,
        bytes_per_elem=1, mlp_variant="gelu")
    hardware = FaceHardware(
        mesh_rows=2, mesh_cols=3,
        local_hbm_capacity_bytes=1_000_000_000,
        local_hbm_bandwidth_gbps=1.0, d2d_bandwidth_gbps=2.0,
        peak_perf_tflops=1.0, d2d_latency_ns=0, local_hbm_latency_ns=0)
    specs = tuple(
        FaceInstanceSpec(f"ins{i}", str(i + 1),
                         tuple(i + row * 3 for row in range(2)))
        for i in range(3))
    topology = build_instances(hardware, specs)
    manager = KVCacheManager(
        topology, model, category_mode="typed", layer_policy="adaptive",
        pool_bandwidth_gbps=4.0, pool_latency_ns=10)
    manager.prepare_prefill(
        session_id=SID, target_instance_index=0, history_tokens=0,
        trigger_request_id="seed")
    manager.expand_prefill(
        session_id=SID, instance_index=0, context_tokens=CTX,
        trigger_request_id="seed")
    manager.mark_complete(SID, 64)
    manager._evict_suffix(
        manager._sessions[SID], phase="completion", reason="fixture",
        trigger_request_id="seed", layer_start=12)

    scheduler = Sh30OnlineScheduler.__new__(Sh30OnlineScheduler)
    scheduler.topology = topology
    scheduler.hardware = hardware
    scheduler.model = model
    scheduler.kv_manager = manager
    scheduler.p_chunk = 512
    scheduler._joint_horizon = CausalHorizonEstimator(
        cold_start_default_forward_steps=1)
    scheduler._joint_flows = LinkFlowRegistry()
    scheduler._pool_ports = _PoolPortRegistry()
    scheduler._hbm_ports = HbmPortFlowRegistry()
    scheduler._joint_factors = ServiceFactors()
    scheduler._joint_rates = JointHardwareRates.from_gbps(
        noc_link_gbps=hardware.d2d_bandwidth_gbps, pool_port_gbps=50.0,
        local_hbm_gbps=hardware.local_hbm_bandwidth_gbps,
        d2d_latency_ns=int(hardware.d2d_latency_ns), pool_latency_ns=0)
    scheduler._instance_edge_ports = {
        i.index: tuple(sorted({
            manager.nearest_edge(r) for r in i.ranks}))
        for i in topology.instances}
    scheduler.joint_config = parse_joint_config(
        {"JOINT_REMOTE_READ_TILE_BYTES": "auto",
         "JOINT_ALLOWED_ACTIONS": "remote-read"})
    scheduler._joint_mode = scheduler.joint_config.scheduler_mode
    scheduler._prefill_task_cache = {}
    scheduler._decode_task_load_cache = {}
    scheduler._task_load_cache_capacity = _TASK_LOAD_CACHE_CAPACITY
    scheduler._quota_tracker = None
    scheduler._quota_enrolled = set()
    scheduler._link_telemetry_rates = {}
    scheduler._link_telemetry_flow_counts = {}
    scheduler.predictor_cost = PredictorCostAccounting()
    scheduler._telemetry_last_tick_ns = 0
    scheduler._admission_decision_wall_ns_total = 0
    scheduler._admission_decision_wall_ns_max = 0
    scheduler._admission_decision_count = 0
    scheduler._quota_verdict_wall_ns_total = 0
    scheduler._quota_verdict_candidate_checks = 0
    scheduler._joint_action_selection_counts = {
        "stay": 0, "copy": 0, "remote-read": 0, "recompute_elected": 0,
        "recompute_forced_no_history": 0,
        "recompute_forced_action_mask": 0,
        "recompute_forced_quota_deferred": 0,
        "recompute_forced_evicted_permanent": 0}
    scheduler._snapshot_verify = False
    scheduler._kv_ledger_epoch = 0
    scheduler._ready_frontier = set()
    scheduler.instances = [
        _OnlineInstanceState(index=i) for i in range(3)]
    scheduler._train_max_iter = 5
    scheduler.runtime_by_request_id = {}
    scheduler.online_log_rows = []
    scheduler.online_log_count = 0
    scheduler.decision_log_sink = None
    graph_config = SimpleNamespace(
        npus_count=6, remote_operand_loads=False,
        trace_granularity="request_aggregated",
        inference_groups=[
            SimpleNamespace(ranks=list(i.ranks), pg_name=str(i.index + 1))
            for i in topology.instances],
        layers=16, hidden_size=4, ffn_size=4, vocab_size=4,
        bytes_per_elem=1, num_heads=2, mlp_variant="gelu",
        request_queue=[SimpleNamespace(
            session_arrival_time_ns=0, inter_request_interval_ns=None)])
    scheduler.graph = GraphBatchBuilder(graph_config)
    scheduler.graph.begin_batch()
    scheduler.graph.set_plan_resolver(
        lambda rid: scheduler.runtime_by_request_id[rid].plan_dict())
    scheduler.graph.set_next_plan({RID: None})
    scheduler.admission_plans = []
    scheduler.graph.emit_admission_batch = (
        lambda plan: (scheduler.admission_plans.append(plan),
                      {"eviction_watches": []})[1])
    scheduler.ledger_admitted = {}
    scheduler.ledger_issued = {}
    scheduler._emitted_by_delivery = {0: {"requests": []}}
    scheduler._batch = {"assignments": [], "watches": [], "future_alarms": [],
                        "delivery_sequence": 0, "tick": 0}
    scheduler.train_ledger_rows = []
    scheduler.train_ledger_sink = None
    scheduler._train_instance_index = {}
    scheduler._pending_merge_alarms = {}
    scheduler._stalled_by_instance = {
        s.index: set() for s in scheduler.instances}
    scheduler.completed_requests = 0
    scheduler.next_request = {RID: None}
    scheduler.runtimes = []
    scheduler._runtime_index = {}
    scheduler.config = SimpleNamespace(request_queue=[
        SimpleNamespace(inter_request_interval_ns=None,
                        next_trigger_type=None)])
    scheduler._eviction_watch_seq = {}
    scheduler._provisional_kv_actions = {}
    scheduler._acked_through = -1
    scheduler._acked_out_of_order = set()
    scheduler.last_applied_sequence = -1
    scheduler.ack_count = 0
    scheduler.delivery_count = 0
    scheduler.committed_kv_action_batches = 0
    scheduler.ledger_committed = {}
    scheduler._delivery_reply_cache = None
    scheduler.in_flight = {}
    scheduler.propagating_tail = PropagatingTailTracker()
    scheduler.propagating_tail.register(
        _PROPAGATING_TAIL_SOURCE_ARRIVALS, lambda: scheduler.in_flight, 4)
    scheduler.propagating_tail.register(
        _PROPAGATING_TAIL_SOURCE_DELIVERY,
        lambda: scheduler._emitted_by_delivery, 8)
    scheduler.propagating_tail.register(
        _PROPAGATING_TAIL_SOURCE_KV,
        lambda: scheduler._provisional_kv_actions, 8)
    scheduler.digest_sink = None
    scheduler.profile_sink = None
    scheduler._profile_batch = {"scanned_entries": 0}

    runtime = sh30._OnlineRequestRuntime({
        "request_id": RID, "session_id": SID, "turn_index": 1,
        "queue_index": 0, "prefill_length": INPUT_TOKENS,
        "decode_length": DECODE, "history_tokens_before": CTX,
        "prefill_context_tokens": CTX + INPUT_TOKENS,
        "final_context_tokens": CTX + INPUT_TOKENS + DECODE,
    }, 512)
    runtime.estimated_arrival_ns = 0

    from online.online_scheduler_base import SCHEMA_VERSION

    def start_delivery(tick, seq):
        scheduler._start_batch({
            "schema_version": 1, "delivery_sequence": seq,
            "delivery_epoch": seq, "tick": tick,
            "deferred_from_tick": tick, "reasons": ["ARRIVAL"],
            "arrivals": [], "completed_groups": [], "completed_nodes": [],
            "affected_ranks": [], "snapshot_handle": {
                "epoch": seq, "tick": tick}})

    def ack(seq):
        scheduler.last_applied_sequence = seq
        scheduler.on_commit_ack({
            "schema_version": SCHEMA_VERSION,
            "delivery_sequence": seq, "batch_id": seq})

    p_record = SelectionRecord(
        mode="joint",
        chosen=ActionCandidate(
            instance_index=1, action="remote-read", applicable=True,
            inapplicable_reason=None, cost_ns=1000),
        candidates=(ActionCandidate(
            instance_index=1, action="remote-read", applicable=True,
            inapplicable_reason=None, cost_ns=1000),),
        instance_rule_note="forced P remote-read", remote_enabled=True)
    d_record = SelectionRecord(
        mode="joint",
        chosen=ActionCandidate(
            instance_index=2, action="remote-read", applicable=True,
            inapplicable_reason=None, cost_ns=1000),
        candidates=(ActionCandidate(
            instance_index=2, action="remote-read", applicable=True,
            inapplicable_reason=None, cost_ns=1000),),
        instance_rule_note="forced multi-source D", remote_enabled=True)

    op = sh30.select_instance_and_action
    sh30.select_instance_and_action = lambda **kw: p_record
    try:
        admitted = scheduler._try_admit_request(runtime, 500)
    finally:
        sh30.select_instance_and_action = op
    assert admitted
    scheduler.runtime_by_request_id[RID] = runtime
    scheduler.runtimes.append(runtime)
    scheduler._runtime_index[RID] = 0

    start_delivery(600, 0)
    scheduler._plan_and_emit_trains(600)
    OnlineSchedulerBase.build_graph_batch(scheduler)
    ack(0)
    plan = scheduler.instances[1].in_flight_train
    scheduler._finalize_completed_trains(
        tuple(plan["drain_members"]), (),
        tuple([plan["train_id"]] if plan["sentinel"] else []), 900)

    od = sh30.select_decode_target_and_action
    sh30.select_decode_target_and_action = lambda **kw: d_record
    try:
        scheduler._on_prefill_drain(RID, 1000)
    finally:
        sh30.select_decode_target_and_action = od

    start_delivery(1100, 1)
    scheduler._plan_and_emit_trains(1100)
    d_batch = OnlineSchedulerBase.build_graph_batch(scheduler)
    ack(1)
    return scheduler, manager, runtime, d_batch


class GrowthReservationHoldTest(unittest.TestCase):
    def test_multi_source_d_completes_naturally_with_stream_fed_suffix(self):
        """正例（修复后的物理口径）：两源 D@第三目标全序列自然完成。

        足迹口径：基础后缀已由 P 相恢复在 i_P 工作副本（读流全层供
        给），D 阶段目标侧后缀承诺为零——drain 点预约直接注销（守卫
        不触发）；j 侧工作副本增量口径物化（无重复后缀）；轮末三位置
        合并守恒通过；lease 生命周期正常（在册→完成注销）。
        """
        scheduler, manager, runtime, d_batch = _drive()
        self.assertIsNotNone(scheduler.instances[2].in_flight_train)
        self.assertEqual(len(d_batch["pipeline_descriptors"]), 2)
        self.assertTrue(all(
            wire["stream_kind"] == "decode_remote_credit"
            for wire in d_batch["pipeline_descriptors"]))
        # drain 点：growth-only 形状的预约已直接注销（后缀承诺为零，
        # extra 为空——hold 不需要介入）。
        self.assertFalse(runtime.joint_growth_reservation_hold)
        self.assertNotIn(RID, manager._reservations)
        # 读流 lease 在册（staging→lease 转换正常发生）。
        self.assertIsNotNone(manager.staging_lease_for_request(RID))
        # decode 至自然完成（三位置合并守恒在 _complete_requests 内
        # 核验——异常即失败）。
        tick = 1200
        seq = 1
        while (not runtime.completed
               and runtime.decode_tokens_consumed < DECODE
               and seq < 12):
            state = next(
                (s for s in scheduler.instances
                 if s.in_flight_train is not None), None)
            plan = state.in_flight_train if state else None
            if plan is None:
                break
            remaining = DECODE - runtime.decode_tokens_consumed
            completed = (RID,) if remaining <= 5 else ()
            scheduler._finalize_completed_trains(
                tuple(plan["drain_members"]), completed,
                tuple([plan["train_id"]] if plan["sentinel"] else []),
                tick + 100)
            if completed:
                scheduler._complete_requests([RID], tick + 200)
                break
            tick += 300
            seq += 1
            scheduler._start_batch({
                "schema_version": 1, "delivery_sequence": seq,
                "delivery_epoch": seq, "tick": tick,
                "deferred_from_tick": tick, "reasons": ["ARRIVAL"],
                "arrivals": [], "completed_groups": [],
                "completed_nodes": [], "affected_ranks": [],
                "snapshot_handle": {"epoch": seq, "tick": tick}})
            scheduler._plan_and_emit_trains(tick)
            OnlineSchedulerBase.build_graph_batch(scheduler)
        self.assertTrue(runtime.completed)
        # 会话随终轮完成结算退休；lease 已随完成注销（无泄漏）。
        self.assertNotIn(SID, manager._sessions)
        self.assertIsNone(manager.staging_lease_for_request(RID))

    def test_suffix_promise_zero_when_working_copy_holds_suffix(self):
        """足迹口径半边：P 相 remote-read 执行副本（第三实例）已持基础
        后缀时，D 阶段目标侧后缀承诺为零（读流供给，不重复恢复）。"""
        from face_scheduler import kv_cache_shard_bytes_for_layer_range
        scheduler, manager, runtime, _ = _drive()
        session = manager._sessions[SID]
        # drain 后（边界事务已提交）：boundary source 记录在 i_P。
        self.assertIsNotNone(session.boundary_source_instance_index)
        suffix, suffix_tokens, suffix_start = (
            manager._partial_remote_read_suffix(
                session=session, history_tokens=178,
                action="remote-read", target_instance_index=2))
        self.assertEqual(suffix, (0, 0))
        self.assertIsNone(suffix_tokens)

    def test_hold_engages_for_genuine_suffix_promise(self):
        """hold 机制镜像（单元面）：预约真实携带未物化 extra 时，
        _convert_drain_staging_to_lease 置 hold 并承接 lease（不提前
        释放）；物化前守卫仍拒绝直接释放。"""
        scheduler, manager, runtime, _ = _drive()
        # 已在 drain 正常释放（本形状 extra 为空）——重新构造一个带
        # 未物化后缀分量的预约 + growth-only 上下文，直接驱动转换器。
        from face_scheduler import KVCapacityReservation
        plan = runtime.remote_read_credit_plan
        self.assertIsNotNone(plan)
        manager._reservations[RID] = KVCapacityReservation(
            request_id=RID, session_id=SID, instance_index=2,
            final_context_tokens=0, final_shard_bytes=(0, 0),
            action="remote-read",
            suffix_history_tokens=128, suffix_start_layer=12,
            staging_shard_bytes=(), staging_tile_tokens=0)
        runtime.joint_remote_growth_only = True
        scheduler._convert_drain_staging_to_lease(runtime, 2000)
        self.assertTrue(runtime.joint_growth_reservation_hold)
        self.assertIn(RID, manager._reservations)
        self.assertIsNotNone(manager.staging_lease_for_request(RID))
        # 物化前直接释放仍被守卫拒绝（真实泄漏不被掩盖）。
        with self.assertRaisesRegex(
                RuntimeError, "before final KV allocation"):
            manager.release_request_capacity_reservation(RID)
        self.assertIn(RID, manager._reservations)


if __name__ == "__main__":
    unittest.main()
