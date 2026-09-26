#!/usr/bin/env python3
"""test_prefill_remote_read_lifecycle.py -- 规格书第六节第 3、5 组回归
（decode 复用与流生命周期；2026-09-25 prefill remote-read 分阶段）。

被测语义（PARTIAL 基，exec != home，真实 KVCacheManager + 真实
GraphBatchBuilder 列车发射 + 真实完成/合并链）：

  组 3（decode 复用）：
    - drain 后只存在前缀 [0, p) credit（计划层界 / readplan 单位流 /
      逐列车切片三处同口径）；
    - 不产生 suffix remote_load（后缀已在准入相池恢复物化为热 KV，
      decode 期 restore journal 关账、池端口零登记、恢复事件零新增）；
    - decode compute 能消费 exec HBM 中的 suffix（增量纯本地生长叠在
      已恢复后缀上；merge v2 以"后缀历史 + 本轮增量"的 exec 侧真值
      参与少并多裁决并胜出）。

  组 5（流生命周期）：
    - rid#prefill_read 准入登记（noc_migrate 双端点 HBM 端口 + 链路流
      + 池端口零登记）、prefill drain 释放；
    - rid#readplan drain 后建立（_reconcile_readplan_at_drain 真值
      登记）、completion 后释放（_settle_readplan_residual）；
    - 无 owner 泄漏或重复登记：读流家族（#prefill_read / #readplan /
      #decode#{j}）在任一检查点恰一个 owner 在册（同一条前缀读流不得
      同时挂 prefill/decode 两个身份，当前 credit 流与未来承诺流不得
      同时占注册表）；run 尾 _assert_no_readplan_leaks /
      _assert_no_flow_registry_leaks 双审计通过。

夹具边界（与 test_prefill_remote_read_scheduler.py 的 _GraphStub 同
哲学，只替身与本组语义无关的到达握手面）：准入批的 turn 间隔门账本
（emit_admission_batch 的 pending_history 注册/消费）由捕获替身承担
——turn>0 准入在图侧要求上一轮 completion 预注册的 interval gate，
属于到达合同（test_joint_arrival_contract）领域；列车发射、完成批
（含 credit arm / 前缀读流 arm 残留 fail-closed 审计）与 merge 尾标记
全部走真构图。decode 列车按续坐形态（joiner_runtimes=[]，T2+ 生产
路径）发射，joiner readiness barrier 形态由 test_remote_credit_stream
覆盖。配额 off 档（_quota_tracker=None）。

运行：cd sh_test_mesh/workload/llama2_7b_inference &&
      python3 online/test_prefill_remote_read_lifecycle.py
      （或 pytest 同路径）
"""
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
    kv_cache_shard_bytes_for_layer_range,
    kv_cache_shard_bytes_for_tokens,
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
import online.sh30_online_scheduler as sh30  # noqa: E402
from online.graph_batch_builder import GraphBatchBuilder  # noqa: E402
from online.sh30_online_scheduler import (  # noqa: E402
    Sh30OnlineScheduler,
    _OnlineInstanceState,
    _OnlineRequestRuntime,
    _PoolPortRegistry,
    _TASK_LOAD_CACHE_CAPACITY,  # F6 销账：替身对齐 __init__ 初值
)

HOME_INSTANCE = 0   # ranks (0, 1)
EXEC_INSTANCE = 1   # ranks (2, 3)
MODEL_LAYERS = 16
PREFIX_LAYERS = 12  # PARTIAL 基驻留前缀 p（后缀 [12, 16) 已逐出到池）
HISTORY_TOKENS = 100
INPUT_TOKENS = 50
DECODE_TOKENS = 8   # credit_iters="4" + _train_max_iter=5 → T1 五步/2 块 + T2 三步/1 块
CREDIT_ITERS = "4"
RID = "r1"
SID = "s"


def _model():
    return FaceModel(
        layers=MODEL_LAYERS, hidden_size=4, ffn_size=4, num_heads=2,
        vocab_size=4, bytes_per_elem=1, mlp_variant="gelu")


def _make_manager(model):
    hardware = FaceHardware(
        mesh_rows=2, mesh_cols=2,
        local_hbm_capacity_bytes=1_000_000,
        local_hbm_bandwidth_gbps=1.0, d2d_bandwidth_gbps=2.0,
        peak_perf_tflops=1.0, d2d_latency_ns=0, local_hbm_latency_ns=0)
    topology = build_instances(
        hardware,
        (FaceInstanceSpec("ins0", "1", (0, 1)),
         FaceInstanceSpec("ins1", "2", (2, 3))))
    manager = KVCacheManager(
        topology, model, category_mode="typed", layer_policy="adaptive",
        pool_bandwidth_gbps=4.0, pool_latency_ns=10)
    return hardware, topology, manager


def _seed_partial(manager, *, prefix_layers: int = PREFIX_LAYERS) -> None:
    """home（instance 0）生长到 HISTORY 后逐出 [p, L) → PARTIAL 会话。"""
    manager.prepare_prefill(
        session_id=SID, target_instance_index=HOME_INSTANCE,
        history_tokens=0, trigger_request_id="s_seed")
    manager.expand_prefill(
        session_id=SID, instance_index=HOME_INSTANCE,
        context_tokens=HISTORY_TOKENS, trigger_request_id="s_seed")
    manager.mark_complete(SID, HISTORY_TOKENS)
    manager._evict_suffix(
        manager._sessions[SID], phase="completion", reason="fixture",
        trigger_request_id="s_seed", layer_start=prefix_layers)


def _make_scheduler(model, hardware, topology, manager):
    """绕过 __init__ 装配准入→drain→decode 列车→完成全链属性面
    （test_prefill_remote_read_scheduler 同款底盘 + 真实构图器 +
    完成链所需账面）。"""
    scheduler = Sh30OnlineScheduler.__new__(Sh30OnlineScheduler)
    scheduler.topology = topology
    scheduler.hardware = hardware
    scheduler.model = model
    scheduler.kv_manager = manager
    scheduler.p_chunk = 512
    scheduler._joint_horizon = CausalHorizonEstimator(
        cold_start_default_tokens=1)
    scheduler._joint_flows = LinkFlowRegistry()
    scheduler._pool_ports = _PoolPortRegistry()
    scheduler._hbm_ports = HbmPortFlowRegistry()
    scheduler._joint_factors = ServiceFactors()
    scheduler._joint_rates = JointHardwareRates.from_gbps(
        noc_link_gbps=hardware.d2d_bandwidth_gbps,
        pool_port_gbps=50.0,
        local_hbm_gbps=hardware.local_hbm_bandwidth_gbps,
        d2d_latency_ns=int(hardware.d2d_latency_ns),
        pool_latency_ns=0)
    scheduler._instance_edge_ports = {
        instance.index: tuple(sorted({
            manager.nearest_edge(rank) for rank in instance.ranks}))
        for instance in topology.instances}
    scheduler.joint_config = parse_joint_config(
        {"JOINT_REMOTE_CREDIT_ITERS": CREDIT_ITERS})
    scheduler._joint_mode = scheduler.joint_config.scheduler_mode
    scheduler._prefill_task_cache = {}
    scheduler._decode_task_load_cache = {}
    scheduler._task_load_cache_capacity = _TASK_LOAD_CACHE_CAPACITY
    scheduler._quota_tracker = None  # off 档（配额面由 fix1 套件覆盖）
    scheduler._quota_enrolled = set()
    scheduler._link_telemetry_rates = {}
    scheduler._link_telemetry_flow_counts = {}
    scheduler._telemetry_last_tick_ns = 0
    scheduler._admission_decision_wall_ns_total = 0
    scheduler._admission_decision_wall_ns_max = 0
    scheduler._admission_decision_count = 0
    scheduler._quota_verdict_wall_ns_total = 0
    scheduler._quota_verdict_candidate_checks = 0
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
    scheduler.instances = [
        _OnlineInstanceState(index=i) for i in range(len(topology.instances))]
    scheduler.runtime_by_request_id = {}
    # T_max=5 < DECODE=8：decode 拆 T1（哨兵列车，5 步）+ T2（exit 列车，
    # 3 步）——#readplan ↔ #decode#{j} 交替的完整状态机需要两列车。
    scheduler._train_max_iter = 5
    scheduler.online_log_rows = []
    scheduler.online_log_count = 0
    scheduler.decision_log_sink = None
    graph_config = SimpleNamespace(
        npus_count=4, remote_operand_loads=False,
        trace_granularity="request_aggregated",
        inference_groups=[
            SimpleNamespace(ranks=(0, 1), pg_name="tp_prefill"),
            SimpleNamespace(ranks=(2, 3), pg_name="tp_decode")],
        layers=MODEL_LAYERS, hidden_size=4, ffn_size=4, vocab_size=4,
        bytes_per_elem=1, num_heads=2, mlp_variant="gelu",
        request_queue=[SimpleNamespace(
            session_arrival_time_ns=0, inter_request_interval_ns=None)])
    scheduler.graph = GraphBatchBuilder(graph_config)
    scheduler.graph.begin_batch()
    scheduler.graph.set_plan_resolver(
        lambda request_id: (
            scheduler.runtime_by_request_id[request_id].plan_dict()))
    # RID = 终轮（无下一 turn）：完成批的 next_plan 消费面（None 分支 =
    # 终态会话清账）。
    scheduler.graph.set_next_plan({RID: None})
    # 夹具边界（见文件头）：准入批 turn 握手替身——捕获 plan 供断言，
    # 其余图面（列车/完成批/merge 标记）全部真实。
    scheduler.admission_plans = []

    def _stub_admission_batch(plan):
        scheduler.admission_plans.append(plan)
        return {"eviction_watches": []}

    scheduler.graph.emit_admission_batch = _stub_admission_batch
    scheduler.ledger_admitted = {}
    scheduler.ledger_issued = {}
    scheduler._emitted_by_delivery = {0: {"requests": []}}
    scheduler._batch = {
        "assignments": [], "watches": [], "future_alarms": [],
        "delivery_sequence": 0, "tick": 0}
    scheduler.train_ledger_rows = []
    scheduler.train_ledger_sink = None
    scheduler._train_instance_index = {}
    scheduler._pending_merge_alarms = {}
    scheduler._stalled_by_instance = {
        state.index: set() for state in scheduler.instances}
    scheduler.completed_requests = 0
    scheduler.next_request = {RID: None}
    scheduler.runtimes = []
    scheduler._runtime_index = {}
    scheduler.config = SimpleNamespace(request_queue=[
        SimpleNamespace(inter_request_interval_ns=None,
                        next_trigger_type=None)])
    scheduler._eviction_watch_seq = {}
    return scheduler


def _make_runtime():
    runtime = _OnlineRequestRuntime({
        "request_id": RID,
        "session_id": SID,
        "turn_index": 1,
        "queue_index": 0,
        "prefill_length": INPUT_TOKENS,
        "decode_length": DECODE_TOKENS,
        "history_tokens_before": HISTORY_TOKENS,
        "prefill_context_tokens": HISTORY_TOKENS + INPUT_TOKENS,
        "final_context_tokens": (
            HISTORY_TOKENS + INPUT_TOKENS + DECODE_TOKENS),
    }, 512)
    runtime.estimated_arrival_ns = 0
    return runtime


def _force_remote_read(exec_instance=EXEC_INSTANCE):
    """select_instance_and_action 打桩：强制 remote-read@exec（被测机器
    = 流生命周期，非 argmin；test_prefill_remote_read_scheduler 同款）。"""
    fake_record = SelectionRecord(
        mode="joint",
        chosen=ActionCandidate(
            instance_index=exec_instance, action="remote-read",
            applicable=True, inapplicable_reason=None, cost_ns=1000),
        candidates=(
            ActionCandidate(
                instance_index=exec_instance, action="remote-read",
                applicable=True, inapplicable_reason=None, cost_ns=1000),),
        instance_rule_note="forced remote-read scene",
        remote_enabled=True)
    original_select = sh30.select_instance_and_action
    sh30.select_instance_and_action = lambda **kw: fake_record
    return original_select


def _kv_bytes(manager, instance_index):
    return tuple(
        manager._rank_states[rank].kv_cache_bytes
        for rank in manager.topology.instance(instance_index).ranks)


def _hbm_owner_ports(scheduler, owner):
    return scheduler._hbm_ports.leaked_owners().get(owner)


def _read_stream_owners(scheduler):
    """读流家族在 HBM 端口注册表的在册 owner 快照（恰一 owner 纪律的
    状态机观测面；链路注册表同源同生命周期，取一即可代表双表）。"""
    owners = scheduler._hbm_ports.leaked_owners()
    return sorted(
        owner for owner in owners
        if owner.endswith("#prefill_read")
        or owner.endswith("#readplan")
        or owner.startswith(RID + "#decode#"))


class _LifecycleRun:
    """一轮完整生命周期 drive + 逐边界检查点快照（owner 状态机 + KV
    真值 + 决策行），供组 3 / 组 5 两套断言共用。"""

    def __init__(self):
        self.model = _model()
        self.hardware, self.topology, self.manager = _make_manager(
            self.model)
        _seed_partial(self.manager)
        self.scheduler = _make_scheduler(
            self.model, self.hardware, self.topology, self.manager)
        self.runtime = _make_runtime()
        self.checkpoints = {}
        # drain 对账时点捕获：#prefill_read 必须先释放、#readplan 后建立
        # （switchover 次序 = 组 5 的核心时序断言）。
        self._reconcile_entry_owners = []
        original_reconcile = self.scheduler._reconcile_readplan_at_drain

        def spy_reconcile(runtime, tick):
            self._reconcile_entry_owners.append({
                "hbm": sorted(self.scheduler._hbm_ports.leaked_owners()),
                "pool": sorted(self.scheduler._pool_ports.leaked_owners()),
            })
            original_reconcile(runtime, tick)

        self.scheduler._reconcile_readplan_at_drain = spy_reconcile

    # ------------------------------------------------------------ 字节 --
    def prefix_bytes_at(self, tokens):
        return kv_cache_shard_bytes_for_layer_range(
            self.model, tokens, self.manager.tp_degree,
            layer_start=0, layer_end=PREFIX_LAYERS)

    def suffix_bytes_at(self, tokens):
        return kv_cache_shard_bytes_for_layer_range(
            self.model, tokens, self.manager.tp_degree,
            layer_start=PREFIX_LAYERS, layer_end=MODEL_LAYERS)

    def full_bytes_at(self, tokens):
        return kv_cache_shard_bytes_for_tokens(
            self.model, tokens, self.manager.tp_degree)

    # ------------------------------------------------------------ 各相 --
    def admit(self):
        original_select = _force_remote_read()
        try:
            admitted = self.scheduler._try_admit_request(self.runtime, 500)
        finally:
            sh30.select_instance_and_action = original_select
        assert admitted
        self.scheduler.runtime_by_request_id[RID] = self.runtime
        self.scheduler.runtimes.append(self.runtime)
        self.scheduler._runtime_index[RID] = 0
        # 恢复事件基线（准入相后缀恢复 issue 恰一次；decode 期不得新增）。
        self.restore_events_at_admission = len(self.manager.restore_events)
        self.checkpoints["admission"] = self._snapshot()

    def drain(self):
        # 列车核销旁路（不拼 prefill 列车）：drain 的 fail-closed 断言
        # 要求聚合账本残值为 0（生产由列车核销精确扣减）。
        self.runtime.queued_chunk_load_ns = 0
        self.scheduler._on_prefill_drain(RID, 1000)
        self.checkpoints["drain"] = self._snapshot()
        # readplan 单位流模板（对账真值登记的注册原料；完成边界 preplan
        # 置 None，此处不捕获即无处可断言）。
        self.readplan_unit_transfers = tuple(
            self.runtime.remote_read_preplan["unit_transfers"])

    def _emit_and_settle_train(self, tick, settle_tick, completed_now):
        """续坐形态 decode 列车（joiner_runtimes=[]）发射 + 核销。"""
        state = self.scheduler.instances[EXEC_INSTANCE]
        if self.runtime in state.pending_decode_ready:
            # 生产由 _plan_and_emit_trains 完成 pending→active 迁移；
            # 续坐形态直接入批（迁移随加入列车发射的 barrier 形态不在
            # 本组语义内，见文件头）。
            state.pending_decode_ready.remove(self.runtime)
            state.active_decode.append(self.runtime)
            state.active_decode_lookup.add(self.runtime)
        plan = self.scheduler._plan_train(state)
        self.scheduler._emit_train(state, plan, [], tick)
        self.checkpoints["emit@%d" % tick] = self._snapshot()
        drained = tuple(plan["drain_members"])
        sentinel = [plan["train_id"]] if plan["sentinel"] else []
        self.scheduler._finalize_completed_trains(
            drained, list(completed_now), tuple(sentinel), settle_tick)
        self.checkpoints["settle@%d" % settle_tick] = self._snapshot()
        return plan

    def run_decode_trains(self):
        # T1：5 步（T_max 截断，哨兵信号核销）；T2：3 步（exit 核销）。
        self.plan1 = self._emit_and_settle_train(2000, 3000, ())
        self.plan2 = self._emit_and_settle_train(3500, 4000, (RID,))

    def complete_and_merge(self):
        self.exec_bytes_pre_merge = _kv_bytes(self.manager, EXEC_INSTANCE)
        self.home_bytes_pre_merge = _kv_bytes(self.manager, HOME_INSTANCE)
        self.scheduler._complete_requests([RID], 5000)
        self.checkpoints["completion"] = self._snapshot()
        self.scheduler._on_merge_done("batch_train_merge_" + RID, 6000)
        self.checkpoints["merge_done"] = self._snapshot()

    # ------------------------------------------------------------ 快照 --
    def _snapshot(self):
        scheduler = self.scheduler
        # 终轮完成边界 retire_terminal_session 注销会话——完成后各检查
        # 点的会话字段落 None（KV 真值断言只用于完成前的检查点）。
        session = self.manager._sessions.get(SID)
        preplan = self.runtime.remote_read_preplan
        return {
            "hbm_owners": dict(scheduler._hbm_ports.leaked_owners()),
            "pool_owners": dict(scheduler._pool_ports.leaked_owners()),
            "link_owners": dict(scheduler._joint_flows.leaked_owners()),
            "read_family": _read_stream_owners(scheduler),
            "preplan_active": (
                None if preplan is None else preplan["registry_active"]),
            "preplan_est_units": (
                None if preplan is None else preplan["est_flow_units"]),
            "preplan_consumed": (
                None if preplan is None else preplan["consumed_units"]),
            "preplan_remaining": (
                None if preplan is None else preplan["remaining_units"]),
            "session_context": (
                None if session is None else session.context_tokens),
            "session_shards": (
                None if session is None else session.shard_bytes),
            "session_location": (
                None if session is None else session.location),
            "session_working_kind": (
                None if session is None else session.working_kind),
            "session_prefix_layers": (
                None if session is None else session.resident_prefix_layers),
            "journal_open": (
                session is not None
                and session.restore_journal is not None),
            "decode_consumed": self.runtime.decode_tokens_consumed,
        }

    def _log_rows(self, kind):
        return [row for row in self.scheduler.online_log_rows
                if row["kind"] == kind]


def _assert_prefix_only_ranges(test, transfers):
    """读流层区间恒 [0, p)（组 3：decode 只读前缀的传输面锚）。"""
    test.assertTrue(transfers)
    for transfer in transfers:
        test.assertEqual(transfer.layer_start, 0)
        test.assertEqual(transfer.layer_end, PREFIX_LAYERS)


class DecodeReuseTest(unittest.TestCase):
    """组 3：drain 后只存在前缀 [0, p) credit；不产生 suffix
    remote_load；decode compute 消费 exec HBM 已恢复后缀。"""

    def setUp(self):
        self.run = _LifecycleRun()
        self.run.admit()
        self.run.drain()
        self.run.run_decode_trains()
        self.run.complete_and_merge()

    def test_decode_credit_covers_prefix_only(self):
        # §四.1/§四.2：持久读计划层界 = p（PARTIAL 混合基，严格小于 L
        # ——构造性排除"credit 覆盖全层"回归）。
        plan = self.run.runtime.remote_read_credit_plan
        self.assertIsNotNone(plan)
        self.assertEqual(plan["home_instance"], HOME_INSTANCE)
        self.assertEqual(plan["exec_instance"], EXEC_INSTANCE)
        self.assertEqual(plan["read_prefix_layers"], PREFIX_LAYERS)
        self.assertLess(plan["read_prefix_layers"], MODEL_LAYERS)
        # 计划字节 = 前缀层区间派生 × 步数，且严格异于全层口径
        # （构造验证：两口径在本夹具下逐字节可分，非肉眼比对）。
        context_per_step = HISTORY_TOKENS + INPUT_TOKENS + DECODE_TOKENS
        expected_per_step = sum(self.run.prefix_bytes_at(context_per_step))
        full_per_step = sum(self.run.full_bytes_at(context_per_step))
        self.assertNotEqual(expected_per_step, full_per_step)
        self.assertEqual(
            plan["total_bytes"], expected_per_step * DECODE_TOKENS)
        # §四.4：decode 后缀层不经 credit——credit 路径源/目的恒为
        # home/exec 实例 ranks（noc_migrate XY 读腿；无池 leg 的
        # edge_rank 断言见 test_slices_are_transient_prefix_streams）。
        self.assertEqual(
            {spec[0] for spec in plan["shard_specs"]},
            {0, 1})
        self.assertEqual(
            {spec[1] for spec in plan["shard_specs"]},
            {2, 3})
        # readplan 单位流（#readplan 注册模板）与逐列车切片同口径。
        _assert_prefix_only_ranges(self, self.run.readplan_unit_transfers)
        for unit in self.run.readplan_unit_transfers:
            self.assertEqual(unit.kind, "noc_migrate")
            self.assertEqual(unit.phase, "decode")
            self.assertEqual(unit.reason, "remote_read_readplan")
            self.assertEqual(unit.source_instance_index, HOME_INSTANCE)
            self.assertEqual(unit.target_instance_index, EXEC_INSTANCE)
        # I1：Σ逐列车切片 ≡ 计划总量（切片只切步数，不改层界/每步字节）。
        self.assertEqual(
            sum(summary["total_bytes"]
                for summary in self.run.runtime.remote_read_slice_summaries),
            plan["total_bytes"])
        # 两列车切片形状（T_max=5 截断 → T1 五步 2 块 [4,1] / T2 三步
        # 1 块 [3]，K 取列车统一值 min(4, S_j)）。
        summaries = self.run.runtime.remote_read_slice_summaries
        self.assertEqual(
            [(s["steps"], s["block_steps"]) for s in summaries],
            [(5, [4, 1]), (3, [3])])

    def test_slices_are_transient_prefix_streams(self):
        # §一.7/§四：decode credit 切片 = 瞬时读流（不物化任何持久账本），
        # 层区间 = [0, p)、驻留前后恒 p（镜像 prefill 前缀读流口径）。
        blocks = self.run.scheduler._joint_remote_read_slice(
            self.run.runtime, 4, 2)
        self.assertEqual(len(blocks), 2)
        for block in blocks:
            self.assertEqual(block.kind, "noc_migrate")
            self.assertEqual(block.phase, "decode")
            self.assertEqual(block.reason, "remote_read_stream")
            _assert_prefix_only_ranges(self, (block,))
            self.assertEqual(
                (block.resident_prefix_layers_before,
                 block.resident_prefix_layers_after),
                (PREFIX_LAYERS, PREFIX_LAYERS))
            self.assertTrue(
                all(shard.edge_rank is None for shard in block.shards))

    def test_no_suffix_remote_load_after_admission(self):
        runtime = self.run.runtime
        # 准入相后缀腿恰一次：history_transfers 全部 remote_load、覆盖
        # [p, L)（非 stream_only 的池恢复持久腿）——本轮唯一的后缀恢复。
        self.assertTrue(runtime.history_transfers)
        for transfer in runtime.history_transfers:
            self.assertEqual(transfer.kind, "remote_load")
            self.assertFalse(transfer.stream_only)
        ranges = [(t.layer_start, t.layer_end)
                  for t in runtime.history_transfers]
        self.assertEqual(ranges, [(PREFIX_LAYERS, MODEL_LAYERS)])
        # drain 后 restore journal 已消费关账（§四.3 守卫的实际兑现），
        # 且 decode 全程（两列车 + 完成 + 合并）零重开——后缀不二次恢复。
        for name in ("drain", "emit@2000", "settle@3000", "emit@3500",
                     "settle@4000", "completion", "merge_done"):
            self.assertFalse(
                self.run.checkpoints[name]["journal_open"], name)
        # 池端口注册表 decode 期零登记（remote_load 才占池端口；前缀
        # credit/merge 均为 noc 腿）——drain 起家族 owner 全不在册。
        for name in ("drain", "emit@2000", "settle@3000", "emit@3500",
                     "settle@4000", "completion", "merge_done"):
            self.assertEqual(
                self.run.checkpoints[name]["pool_owners"], {}, name)
        # 恢复事件零新增（C15 issue 只在准入相发生一次）。
        self.assertEqual(
            len(self.run.manager.restore_events),
            self.run.restore_events_at_admission + 1)

    def test_decode_grows_locally_on_restored_suffix(self):
        # 准入相（prepare 事务后）：exec HBM 物理真值 = 后缀 S（D1 两口
        # 径分离：context = 增量 0 起步、驻留前缀推进到 L 的工作形态）。
        admission = self.run.checkpoints["admission"]
        self.assertEqual(
            admission["session_shards"],
            self.run.suffix_bytes_at(HISTORY_TOKENS))
        self.assertEqual(admission["session_context"], 0)
        self.assertEqual(admission["session_location"], "local_hbm")
        self.assertEqual(admission["session_working_kind"], "remote-read")
        self.assertEqual(admission["session_prefix_layers"], MODEL_LAYERS)
        # drain：工作上下文 = 已到达输入（增量物化起点）。
        self.assertEqual(
            self.run.checkpoints["drain"]["session_context"],
            INPUT_TOKENS)
        # decode 逐列车因果生长：增量纯本地叠在已恢复后缀上——shard
        # 真值 = 后缀@HISTORY + 全层@当前上下文（无池恢复参与）。
        for name, consumed in (("settle@3000", 5), ("settle@4000", 8)):
            point = self.run.checkpoints[name]
            self.assertEqual(point["decode_consumed"], consumed)
            self.assertEqual(
                point["session_context"], INPUT_TOKENS + consumed)
            self.assertEqual(
                point["session_shards"],
                tuple(a + b for a, b in zip(
                    self.run.suffix_bytes_at(HISTORY_TOKENS),
                    self.run.full_bytes_at(INPUT_TOKENS + consumed))))
        # merge v2 少并多：exec 侧真值（后缀历史 + 本轮增量）> home 前缀
        # → 翻转（winner=exec、home 迁移、传输字节 = home 前缀）——decode
        # compute 的产物与已恢复后缀同账结算，后缀复用贯穿到轮末。
        prefix_sum = sum(self.run.prefix_bytes_at(HISTORY_TOKENS))
        exec_sum = sum(self.run.exec_bytes_pre_merge)
        home_sum = sum(self.run.home_bytes_pre_merge)
        self.assertEqual(home_sum, prefix_sum)
        self.assertGreater(exec_sum, home_sum)
        outcome = self.run.runtime.merge_outcome
        self.assertIsNotNone(outcome)
        self.assertEqual(outcome["direction"], "reverse")
        self.assertEqual(outcome["winner_instance"], EXEC_INSTANCE)
        self.assertTrue(outcome["home_flipped"])
        self.assertEqual(outcome["transferred_bytes"], home_sum)
        row = self.run.manager.kv_delta_find(RID)
        self.assertIsNotNone(row)
        self.assertEqual(row["direction"], "reverse")
        self.assertEqual(row["exec_side_retained_bytes"], exec_sum)


class FlowLifecycleTest(unittest.TestCase):
    """组 5：#prefill_read 准入登记/drain 释放；#readplan drain 后建立/
    completion 后释放；读流家族恰一 owner、无泄漏无重复登记。"""

    def setUp(self):
        self.run = _LifecycleRun()
        self.run.admit()
        self.run.drain()
        self.run.run_decode_trains()
        self.run.complete_and_merge()

    def test_prefill_read_registered_at_admission(self):
        point = self.run.checkpoints["admission"]
        # HBM 端口：noc_migrate 双端点腿——home 读端口 (0,1) + exec 写
        # 端口 (2,3)。前缀层组在图上是 stop-and-wait 串行链（组 g+1 链序
        # 后随组 g 的 ack），serial_credit_stream 折叠为单代表流 = 2 shard
        # × 双端点 = 4 条目（与 #readplan/#decode#j 同纪律；除数不随组数
        # 虚涨）。
        self.assertEqual(
            point["hbm_owners"].get(RID + "#prefill_read"),
            (0, 2, 1, 3))
        self.assertIn(RID + "#prefill_read", point["link_owners"])
        # 池端口零登记（前缀读流不经池路径；池上的在册 owner 是后缀
        # remote_load 腿的主链 owner=rid）。
        self.assertNotIn(RID + "#prefill_read", point["pool_owners"])
        self.assertIn(RID, point["pool_owners"])
        # #readplan 准入相零注册表登记（est 账本在、流不在）——prefill/
        # decode 两阶段不得双计并发读流。
        self.assertNotIn(RID + "#readplan", point["hbm_owners"])
        self.assertNotIn(RID + "#readplan", point["link_owners"])
        self.assertFalse(point["preplan_active"])
        self.assertGreater(point["preplan_est_units"], 0)
        # 恰一 owner：读流家族在册者只有 #prefill_read。
        self.assertEqual(point["read_family"], [RID + "#prefill_read"])

    def test_drain_releases_prefill_read_then_readplan_takes_over(self):
        # switchover 次序：对账（#readplan 建立点）入口时刻 #prefill_read
        # 已不在任何注册表（hbm/pool 双表同证）。
        self.assertEqual(len(self.run._reconcile_entry_owners), 1)
        entry = self.run._reconcile_entry_owners[0]
        self.assertNotIn(RID + "#prefill_read", entry["hbm"])
        self.assertNotIn(RID + "#prefill_read", entry["pool"])
        # drain 后：#readplan 真值登记承接 decode credit（单条每 shard
        # 代表流 = 2 shard × 双端点 = 4 条目），#prefill_read 零残留。
        point = self.run.checkpoints["drain"]
        self.assertNotIn(RID + "#prefill_read", point["hbm_owners"])
        self.assertNotIn(RID + "#prefill_read", point["link_owners"])
        self.assertEqual(
            point["hbm_owners"].get(RID + "#readplan"), (0, 2, 1, 3))
        self.assertIn(RID + "#readplan", point["link_owners"])
        self.assertTrue(point["preplan_active"])
        # 主链 owner（后缀池恢复腿）同边界释放。
        self.assertNotIn(RID, point["hbm_owners"])
        self.assertNotIn(RID, point["pool_owners"])
        self.assertEqual(point["read_family"], [RID + "#readplan"])
        # 对账决策行（C8 步骤 2）：真值步数/真值流单位落盘（真值闭式：
        # 列车数 = ceil(8/5) = 2 × 每列车块数 = ceil(5/4) = 2 → 4 单位）。
        rows = self.run._log_rows("readplan_reconcile")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["decision"]["actual_steps"],
                         DECODE_TOKENS)
        self.assertEqual(rows[0]["decision"]["truth_flow_units"], 4)

    def test_readplan_and_decode_owner_alternate_without_overlap(self):
        # T1 发射：当前列车 owner 接管——#readplan 先核销暂挂、
        # #decode#1 后登记（同一前缀读流不得同时挂两个 decode 身份）。
        emit1 = self.run.checkpoints["emit@2000"]
        self.assertNotIn(RID + "#readplan", emit1["hbm_owners"])
        self.assertEqual(
            emit1["hbm_owners"].get(RID + "#decode#1"), (0, 2, 1, 3))
        self.assertEqual(emit1["read_family"], [RID + "#decode#1"])
        self.assertFalse(emit1["preplan_active"])
        # 串行 credit 折叠：T1 两个切片块（[4,1]）同路径只登记一条代表
        # 流（4 条目 = 2 shard × 双端点，非 2 块 × 2 shard 的并发双计）。
        self.assertEqual(len(emit1["hbm_owners"][RID + "#decode#1"]), 4)
        # T1 核销：decode 未完（5 < 8）→ #decode#1 释放、#readplan 恢复
        # 一条未来代表流（承诺账本回填）。
        settle1 = self.run.checkpoints["settle@3000"]
        self.assertNotIn(RID + "#decode#1", settle1["hbm_owners"])
        self.assertIn(RID + "#readplan", settle1["hbm_owners"])
        self.assertEqual(settle1["read_family"], [RID + "#readplan"])
        self.assertTrue(settle1["preplan_active"])
        # T2 发射：再次接管（j=2）；T2 核销：decode 完成（8 == 8）→
        # 无未来工作，#readplan 不再恢复——家族清空。
        emit2 = self.run.checkpoints["emit@3500"]
        self.assertEqual(emit2["read_family"], [RID + "#decode#2"])
        self.assertNotIn(RID + "#readplan", emit2["hbm_owners"])
        settle2 = self.run.checkpoints["settle@4000"]
        self.assertNotIn(RID + "#decode#2", settle2["hbm_owners"])
        self.assertNotIn(RID + "#readplan", settle2["hbm_owners"])
        self.assertEqual(settle2["read_family"], [])
        self.assertFalse(settle2["preplan_active"])
        # 账本核销链：真值 4 单位，逐列车实际消费 3（2+1，列车碎片化）
        # → 残差 1 在完成边界披露（C8 对账核销之二）。
        self.assertEqual(settle2["preplan_consumed"], 3)
        self.assertEqual(settle2["preplan_remaining"], 1)

    def test_completion_releases_readplan_and_run_end_audits_pass(self):
        completion = self.run.checkpoints["completion"]
        # C8 对账核销之三：完成边界清残（preplan 置 None + #readplan
        # 幂等空放）+ 碎片化残差披露行。
        self.assertIsNone(self.run.runtime.remote_read_preplan)
        self.assertNotIn(RID + "#readplan", completion["hbm_owners"])
        self.assertEqual(completion["read_family"], [])
        settle_rows = self.run._log_rows("readplan_settle")
        self.assertEqual(len(settle_rows), 1)
        self.assertEqual(
            settle_rows[0]["decision"]["residual_units_released"], 1)
        self.assertEqual(settle_rows[0]["decision"]["consumed_units"], 3)
        self.assertEqual(settle_rows[0]["decision"]["truth_flow_units"], 4)
        self.assertEqual(settle_rows[0]["decision"]["variance_units"], 1)
        # merge 在途流在册（完成批登记、merge_done 交付前）——家族断言
        # 只覆盖读流家族，merge owner 单独验证生命周期配对。
        self.assertIn(RID + "#merge", completion["hbm_owners"])
        # merge_done 交付：merge 流注销 → 三注册表全空（无任何 owner
        # 泄漏），run 尾双审计通过。
        merged = self.run.checkpoints["merge_done"]
        self.assertEqual(merged["hbm_owners"], {})
        self.assertEqual(merged["pool_owners"], {})
        self.assertEqual(merged["link_owners"], {})
        self.run.scheduler._assert_no_readplan_leaks()
        self.run.scheduler._assert_no_flow_registry_leaks()

    def test_no_duplicate_registration_of_the_same_read_path(self):
        # 重复登记反证（检查点序列逐点断言，任一时刻读路径在册 owner 恰
        # 一个且条目数恒为单代表流口径 4）：
        #   准入 #prefill_read(4=层组串行链折叠单代表流) → drain #readplan(4)
        #   → T1 #decode#1(4) → settle #readplan(4) → T2 #decode#2(4)
        #   → settle ∅ → completion ∅ → merge_done ∅。
        expected_sequence = [
            ("admission", [RID + "#prefill_read"]),
            ("drain", [RID + "#readplan"]),
            ("emit@2000", [RID + "#decode#1"]),
            ("settle@3000", [RID + "#readplan"]),
            ("emit@3500", [RID + "#decode#2"]),
            ("settle@4000", []),
            ("completion", []),
            ("merge_done", []),
        ]
        for name, family in expected_sequence:
            self.assertEqual(
                self.run.checkpoints[name]["read_family"], family, name)


if __name__ == "__main__":
    unittest.main()
