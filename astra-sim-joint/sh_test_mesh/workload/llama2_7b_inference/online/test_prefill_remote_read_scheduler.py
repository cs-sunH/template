#!/usr/bin/env python3
"""test_prefill_remote_read_scheduler.py -- 规格书§二/§四（2026-09-25
prefill remote-read 分阶段）的在线调度器侧回归。

目标语义（PARTIAL 基，exec != home）：prefill 两腿从同一准入 frontier
并行分叉——home 前缀 [0,p) 经 NoC 前缀读流（stream_only 瞬时流）、
remote pool 后缀 [p,L) 经 remote_load 池恢复；decode 只对 home 前缀
[0,p) 发 credit 读流（后缀复用 exec HBM 已恢复副本，不再池恢复）。

覆盖：
  1. runtime 新字段与 history_transfers 严格分离（§二.1/§二.2）：
     准入后 history_transfers 只含后缀池恢复、prefill_remote_read_*
     只含前缀读流；history_transfer_bytes 不含前缀读流字节；
  2. 流生命周期（§二.4）：准入登记 rid#prefill_read（NoC 路径 + home
     HBM 读端口 + exec HBM 写端口、池端口零登记）；prefill drain 释放；
     drain 后才由 #readplan 承接 decode credit——同一条前缀读流不得
     同时挂 prefill/decode 两个身份；
  3. 独立披露字段（§二.3）：joint_admission 决策行 + plan_dict 的
     prefill_remote_read_bytes/layers/transfers 三键；
  4. decode 侧（§四）：drain 时断言 suffix restore journal 已关账才建
     decode credit 计划；PARTIAL 基 credit 计划 read_prefix_layers = p
     （只覆盖 [0,p)）；credit 切片 stream_only=True（§一.7 补前序卡
     登记的偏差项）；
  5. rollback 覆盖新 owner（§二.5）：入册失败回滚后 #prefill_read
     零残留（A10b 面在本文件的准入级全链复验）。

运行：cd sh_test_mesh/workload/llama2_7b_inference &&
      python3 online/test_prefill_remote_read_scheduler.py
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
)
from joint.hbm_port_flow_registry import HbmPortFlowRegistry  # noqa: E402
from joint.pipeline_identity import PipelineIdentityAllocator  # noqa: E402
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
from online.sh30_online_scheduler import (  # noqa: E402
    Sh30OnlineScheduler,
    _OnlineInstanceState,
    _OnlineRequestRuntime,
    _PoolPortRegistry,
    _TASK_LOAD_CACHE_CAPACITY,  # F6 销账：替身对齐 __init__ 初值
)

HOME_INSTANCE = 0   # ranks (0, 1)
EXEC_INSTANCE = 1   # ranks (2, 3)
HISTORY_TOKENS = 100
PREFIX_LAYERS = 12  # PARTIAL 基驻留前缀 p
INPUT_TOKENS = 50
DECODE_TOKENS = 8
RID = "r1"
SID = "s"


def _model(*, layers: int = 16):
    return FaceModel(
        layers=layers, hidden_size=4, ffn_size=4, num_heads=2,
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


def _seed_resident(manager, *, tokens: int = HISTORY_TOKENS) -> None:
    """在 home（instance 0）完成一轮生长到 tokens 的 LOCAL 会话。"""
    manager.prepare_prefill(
        session_id=SID, target_instance_index=HOME_INSTANCE,
        history_tokens=0, trigger_request_id="s_seed")
    manager.expand_prefill(
        session_id=SID, instance_index=HOME_INSTANCE,
        context_tokens=tokens, trigger_request_id="s_seed")
    manager.mark_complete(SID, tokens)


def _seed_partial(manager, *, prefix_layers: int = PREFIX_LAYERS) -> None:
    """完成一轮并逐出 [p, L) → PARTIAL 会话（home 不变）。"""
    _seed_resident(manager)
    manager._evict_suffix(
        manager._sessions[SID], phase="completion", reason="fixture",
        trigger_request_id="s_seed", layer_start=prefix_layers)


class _GraphStub:
    """准入/发射链最小替身：捕获 plan_dict（_emit_admission 消费）。"""

    def __init__(self):
        self.admission_plans = []
        self.synced = []
        self.pipeline_identity_allocator = PipelineIdentityAllocator()

    def emit_admission_batch(self, plan):
        self.admission_plans.append(plan)
        return {"eviction_watches": []}

    def sync_pending_history_after_evictions(self, evictions):
        self.synced.append(evictions)


def _make_scheduler(model, hardware, topology, manager):
    """绕过 __init__（需 manifest/graph/bridge），装配 _try_admit_request
    全链（成功段 + drain 段）用到的属性面（Roofline 小参数，与
    test_joint_decision_schema / test_joint_fix1_pricing 同款）。"""
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
        env={"JOINT_ALLOWED_ACTIONS": "remote-read"})
    scheduler._joint_mode = scheduler.joint_config.scheduler_mode
    scheduler._prefill_task_cache = {}
    scheduler._decode_task_load_cache = {}
    # 对齐 __init__ 初值（F6 销账：软门已删，替身漏设 = AttributeError）。
    scheduler._task_load_cache_capacity = _TASK_LOAD_CACHE_CAPACITY
    scheduler._quota_tracker = None  # off 档（配额面由 fix1 A10b 面覆盖）
    scheduler._link_telemetry_rates = {}
    scheduler._link_telemetry_flow_counts = {}
    # ACC（验收纪律第 3 类，v2.0 §19.3）：预测器开销账本 __init__ 同款
    # 初值（F6 销账口径：决策/摘要记账直访——替身漏设 = AttributeError）。
    from joint.predictor_cost_accounting import PredictorCostAccounting
    scheduler.predictor_cost = PredictorCostAccounting()
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
        "recompute_forced_action_mask": 0,
        "recompute_forced_quota_deferred": 0,
        "recompute_forced_evicted_permanent": 0,
    }
    scheduler._snapshot_verify = False
    scheduler._kv_ledger_epoch = 0
    scheduler._ready_frontier = set()
    scheduler.instances = [
        _OnlineInstanceState(index=i) for i in range(len(topology.instances))]
    scheduler.runtime_by_request_id = {}
    scheduler._train_max_iter = 8
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
    scheduler._task_load_snapshot = (
        lambda state, now_ns: SimpleNamespace(
            queued_prefill_task_load_ns=0,
            running_prefill_task_load_ns=0,
            active_decode_task_load_ns=0,
            ordering_key=(0,)))
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


def _force_remote_read(scheduler, exec_instance=EXEC_INSTANCE):
    """模块级 select_instance_and_action 打桩：强制选中 remote-read@exec
    （被测机器 = 准入事务/登记段与 drain 时序，非 argmin）。"""
    fake_record = SelectionRecord(
        mode="joint",
        chosen=ActionCandidate(
            instance_index=exec_instance, action="remote-read",
            applicable=True, inapplicable_reason=None, cost_ns=1000),
        candidates=(
            ActionCandidate(
                instance_index=HOME_INSTANCE, action="stay",
                applicable=True, inapplicable_reason=None, cost_ns=10),
            ActionCandidate(
                instance_index=exec_instance, action="remote-read",
                applicable=True, inapplicable_reason=None, cost_ns=1000),
        ),
        instance_rule_note="forced remote-read scene",
        remote_enabled=True)
    original_select = sh30.select_instance_and_action
    sh30.select_instance_and_action = lambda **kw: fake_record
    return original_select


def _admission_rows(scheduler, kind):
    return [row for row in scheduler.online_log_rows
            if row["kind"] == kind]


def _layer_ranges(transfers):
    return tuple((t.layer_start, t.layer_end) for t in transfers)


def _assert_contiguous_cover(test, ranges, start: int, end: int) -> None:
    cursor = start
    for layer_start, layer_end in ranges:
        test.assertEqual(layer_start, cursor)
        test.assertGreater(layer_end, layer_start)
        cursor = layer_end
    test.assertEqual(cursor, end)


class PrefillRemoteReadAdmissionTest(unittest.TestCase):
    """§二.1/§二.2/§二.3：PARTIAL 基准入——runtime 字段分离 + 登记 +
    披露字段（真实 KVCacheManager 事务全链）。"""

    def setUp(self):
        self.model = _model()
        self.hardware, self.topology, self.manager = _make_manager(
            self.model)
        _seed_partial(self.manager)
        self.scheduler = _make_scheduler(
            self.model, self.hardware, self.topology, self.manager)
        self.runtime = _make_runtime()
        self.original_select = _force_remote_read(self.scheduler)
        try:
            self.assertTrue(self.scheduler._try_admit_request(
                self.runtime, 500))
        finally:
            sh30.select_instance_and_action = self.original_select
        self.prefix_bytes = kv_cache_shard_bytes_for_layer_range(
            self.model, HISTORY_TOKENS, self.manager.tp_degree,
            layer_start=0, layer_end=PREFIX_LAYERS)
        self.suffix_bytes = kv_cache_shard_bytes_for_layer_range(
            self.model, HISTORY_TOKENS, self.manager.tp_degree,
            layer_start=PREFIX_LAYERS, layer_end=self.model.layers)

    def test_runtime_fields_strictly_separate_two_legs(self):
        runtime = self.runtime
        # 后缀腿：history_transfers 只含 remote_load 逐组恢复（[p, L)）。
        self.assertTrue(runtime.history_transfers)
        for transfer in runtime.history_transfers:
            self.assertEqual(transfer.kind, "remote_load")
            self.assertFalse(transfer.stream_only)
        _assert_contiguous_cover(
            self, _layer_ranges(runtime.history_transfers),
            PREFIX_LAYERS, self.model.layers)
        # 前缀腿（D16 有界模板，修复轮二）：runtime.prefill_remote_read_
        # transfers 恒 ()（tile 数随 U 无界增长，物化移到消费点——列车
        # 规划期经 kv_manager.prefill_read_window_tiles 现算），读流形状
        # 由计划模板 + 逐窗物化承担。
        self.assertIsNotNone(runtime.prefill_remote_read_plan)
        self.assertEqual(runtime.prefill_remote_read_transfers, ())
        plan = runtime.prefill_remote_read_plan
        windows = {
            window: self.manager.prefill_read_window_tiles(
                plan["template"], window)
            for window in range(1, plan["window_count"] + 1)}
        flattened = [
            transfer for window in windows.values() for transfer in window]
        for transfer in flattened:
            self.assertEqual(transfer.kind, "noc_migrate")
            self.assertEqual(transfer.phase, "prefill")
            self.assertEqual(
                transfer.reason, "remote_read_prefill_prefix")
            self.assertTrue(transfer.stream_only)
            self.assertEqual(
                transfer.source_instance_index, HOME_INSTANCE)
            self.assertEqual(
                transfer.target_instance_index, EXEC_INSTANCE)
        _assert_contiguous_cover(
            self, _layer_ranges(flattened), 0, PREFIX_LAYERS)
        # 两腿层区间不相交、并集铺满 [0, L)（同 frontier 分叉语义）。
        self.assertEqual(
            _layer_ranges(runtime.history_transfers)[0][0], PREFIX_LAYERS)
        self.assertEqual(
            _layer_ranges(flattened)[-1][1], PREFIX_LAYERS)
        # 字节口径：前缀读流独立计量，不混入 history_transfer_bytes。
        self.assertEqual(
            runtime.prefill_remote_read_bytes, sum(self.prefix_bytes))
        self.assertEqual(
            runtime.history_transfer_bytes, sum(self.suffix_bytes))
        # 计划摘要锚（home/exec/前缀层界/字节；D16 模板参数）。
        self.assertEqual(plan["home_instance"], HOME_INSTANCE)
        self.assertEqual(plan["exec_instance"], EXEC_INSTANCE)
        self.assertEqual(plan["read_prefix_layers"], PREFIX_LAYERS)
        self.assertEqual(plan["total_bytes"],
                         runtime.prefill_remote_read_bytes)
        # tiles_by_window 与逐窗物化一致；tile_count = Σ逐窗 tile 数。
        self.assertEqual(
            tuple(plan["tiles_by_window"]),
            tuple(len(windows[window])
                  for window in range(1, plan["window_count"] + 1)))
        self.assertEqual(
            plan["tile_count"], len(flattened))
        # D09 逐窗×tile：每窗 tile 恒携带 (窗序, 窗内 tile 序) 身份，
        # 窗序连续 1..W、窗内自 1 连续；窗并集 = 全部传输。
        self.assertEqual(
            tuple(windows), tuple(range(1, plan["window_count"] + 1)))
        for window_index, window in windows.items():
            for position, transfer in enumerate(window, start=1):
                self.assertEqual(
                    transfer.read_tile, (window_index, position))

    def test_prefill_read_registered_and_decode_owner_absent(self):
        # §二.4/§二.5：准入登记 rid#prefill_read（noc_migrate 双端点
        # HBM 端口 + 链路流；池端口零登记——前缀读流不经池路径）；
        # #readplan 注册表半边零登记（est 账本在、流不在——prefill/
        # decode 两阶段不并发双计）。
        hbm_owners = self.scheduler._hbm_ports.leaked_owners()
        self.assertIn(RID + "#prefill_read", hbm_owners)
        self.assertNotIn(RID + "#readplan", hbm_owners)
        # history 的 remote_load 后缀恢复腿为零距离（edge == 目标
        # rank、单 rank noc_path）：修复轮三（2026-09-27）执行侧端点
        # 登记补全后，准入时点即登记目标 HBM 写端口（exec 2,3）——
        # 与计价侧（_pool_transfer_ns_shards）无条件收端点腿同口径；
        # 原断言 "RID 不在册" 钉的是修复前漏登行为，随之更新。
        self.assertEqual(
            sorted(set(hbm_owners[RID])), [2, 3])
        self.assertEqual(
            sorted(set(hbm_owners[RID + "#prefill_read"])),
            [0, 1, 2, 3])  # home 读端口 (0,1) + exec 写端口 (2,3)
        pool_owners = self.scheduler._pool_ports.leaked_owners()
        self.assertNotIn(RID + "#prefill_read", pool_owners)
        # est 承诺账本在场（registry_active=False）。
        preplan = self.runtime.remote_read_preplan
        self.assertIsNotNone(preplan)
        self.assertFalse(preplan["registry_active"])
        self.assertGreater(preplan["est_flow_units"], 0)

    def test_decision_row_and_plan_dict_disclose_new_fields(self):
        # §二.3：joint_admission 行 + plan_dict 三键独立披露。
        # D16 配合项（修复轮二，2026-09-27）：决策行改有界聚合披露
        # ``prefill_remote_read_summary``（物化 transfer 列表退役——tile
        # 数无界；窗口/模板参数足以重建读流形状），plan_dict 快照同样
        # 携带模板参数（tiles_by_window/window_count/tile_bytes/
        # scan_passes/session_id/source_ranks/target_ranks/
        # base_scans_shards——无 "tiles" 物化键）。
        row = _admission_rows(self.scheduler, "joint_admission")[0]
        decision = row["decision"]
        self.assertEqual(
            decision["prefill_remote_read_bytes"],
            self.runtime.prefill_remote_read_bytes)
        self.assertEqual(
            decision["prefill_remote_read_layers"], PREFIX_LAYERS)
        summary = decision["prefill_remote_read_summary"]
        self.assertEqual(
            set(summary), {
                "present", "home_instance", "exec_instance",
                "read_prefix_layers", "total_bytes", "tile_count",
                "window_count", "tiles_by_window", "tile_bytes",
                "scan_passes"})
        self.assertTrue(summary["present"])
        self.assertNotIn("prefill_remote_read_transfers", decision)
        plan = self.runtime.prefill_remote_read_plan
        self.assertEqual(
            summary["tile_count"], plan["tile_count"])
        self.assertEqual(
            summary["tiles_by_window"], plan["tiles_by_window"])
        self.assertEqual(
            summary["total_bytes"],
            self.runtime.prefill_remote_read_bytes)
        self.assertEqual(
            summary["read_prefix_layers"], PREFIX_LAYERS)
        self.assertEqual(summary["home_instance"], HOME_INSTANCE)
        self.assertEqual(summary["exec_instance"], EXEC_INSTANCE)
        self.assertEqual(
            summary["window_count"],
            len(summary["tiles_by_window"]))
        self.assertEqual(
            sum(summary["tiles_by_window"]), summary["tile_count"])
        self.assertEqual(summary["tile_bytes"], plan["tile_bytes"])
        self.assertEqual(summary["scan_passes"], plan["scan_passes"])
        plan_dict = self.scheduler.graph.admission_plans[0]
        self.assertEqual(
            plan_dict["prefill_remote_read_bytes"],
            self.runtime.prefill_remote_read_bytes)
        self.assertEqual(
            plan_dict["prefill_remote_read_layers"], PREFIX_LAYERS)
        # plan_dict 面：物化列表恒空（D16），快照为模板参数形态。
        self.assertEqual(
            plan_dict["prefill_remote_read_transfers"], ())
        snapshot = plan_dict["prefill_remote_read_plan"]
        self.assertEqual(
            set(snapshot), {
                "home_instance", "exec_instance", "read_prefix_layers",
                "total_bytes", "tile_count", "window_count",
                "tiles_by_window", "tile_bytes", "scan_passes",
                "session_id", "source_ranks", "target_ranks",
                "base_scans_shards"})
        self.assertNotIn("tiles", snapshot)
        self.assertEqual(
            snapshot["tiles_by_window"], plan["tiles_by_window"])
        self.assertEqual(snapshot["window_count"], plan["window_count"])
        self.assertEqual(snapshot["tile_bytes"], plan["tile_bytes"])
        self.assertEqual(snapshot["scan_passes"], plan["scan_passes"])
        self.assertEqual(snapshot["session_id"], SID)
        self.assertEqual(snapshot["source_ranks"], (0, 1))
        self.assertEqual(snapshot["target_ranks"], (2, 3))
        self.assertEqual(
            len(snapshot["base_scans_shards"]),
            len(snapshot["source_ranks"]))
        # 分离判据在 plan_dict 面同样成立（图侧分叉铺腿的输入契约）：
        # 前缀读流（noc_migrate）不混入 history_transfers。
        self.assertNotIn(
            "noc_migrate",
            {transfer.kind for transfer in plan_dict["history_transfers"]},
            "prefix read stream must stay out of the history leg")

    def test_local_base_admission_reads_all_layers_without_suffix(self):
        # LOCAL 基回归：p = L——前缀读流覆盖全部层、history_transfers
        # 恒空（无后缀腿）、decode credit 计划仍覆盖全部层。D16：读流
        # 形状经计划模板逐窗物化断言（runtime 物化列表恒空）。
        model = _model()
        _hardware, topology, manager = _make_manager(model)
        _seed_resident(manager)
        scheduler = _make_scheduler(model, _hardware, topology, manager)
        runtime = _make_runtime()
        original_select = _force_remote_read(scheduler)
        try:
            self.assertTrue(scheduler._try_admit_request(runtime, 500))
        finally:
            sh30.select_instance_and_action = original_select
        self.assertEqual(runtime.history_transfers, ())
        self.assertEqual(runtime.history_transfer_bytes, 0)
        self.assertEqual(runtime.prefill_remote_read_transfers, ())
        plan = runtime.prefill_remote_read_plan
        self.assertIsNotNone(plan)
        transfers = [
            transfer
            for window in range(1, plan["window_count"] + 1)
            for transfer in manager.prefill_read_window_tiles(
                plan["template"], window)]
        _assert_contiguous_cover(
            self, _layer_ranges(transfers), 0, model.layers)
        self.assertEqual(
            runtime.prefill_remote_read_plan["read_prefix_layers"],
            model.layers)
        self.assertIn(RID + "#prefill_read",
                      scheduler._hbm_ports.leaked_owners())


class PrefillRemoteReadDrainLifecycleTest(unittest.TestCase):
    """§二.4/§四.3：prefill drain 释放 #prefill_read → 断言 suffix
    restore journal 关账 → 建立 decode credit 计划（#readplan 承接）。"""

    def setUp(self):
        self.model = _model()
        self.hardware, self.topology, self.manager = _make_manager(
            self.model)
        _seed_partial(self.manager)
        self.scheduler = _make_scheduler(
            self.model, self.hardware, self.topology, self.manager)
        self.runtime = _make_runtime()
        self.original_select = _force_remote_read(self.scheduler)
        try:
            self.assertTrue(self.scheduler._try_admit_request(
                self.runtime, 500))
        finally:
            sh30.select_instance_and_action = self.original_select
        self.scheduler.runtime_by_request_id[RID] = self.runtime
        # 列车核销旁路（本测试不拼列车）：drain 的 fail-closed 断言
        # 要求聚合账本残值为 0（生产由列车核销精确扣减到 0）。
        self.runtime.queued_chunk_load_ns = 0
        # 对账时点捕获：#prefill_read 必须已在 #readplan 建立前释放。
        reconcile_owners_at_call = []
        original_reconcile = self.scheduler._reconcile_readplan_at_drain

        def spy_reconcile(runtime_, tick):
            reconcile_owners_at_call.append(
                sorted(self.scheduler._hbm_ports.leaked_owners()))
            original_reconcile(runtime_, tick)

        self.scheduler._reconcile_readplan_at_drain = spy_reconcile
        self.reconcile_owners_at_call = reconcile_owners_at_call

    def test_drain_releases_prefill_read_then_builds_credit_plan(self):
        self.scheduler._on_prefill_drain(RID, 1000)
        # 释放先于建立：对账调用时刻 #prefill_read 已不在册。
        self.assertEqual(len(self.reconcile_owners_at_call), 1)
        self.assertNotIn(
            RID + "#prefill_read", self.reconcile_owners_at_call[0])
        # drain 后：#readplan 承接 decode credit（真值登记），
        # #prefill_read 零残留——同一条前缀读流未同时挂两个身份。
        owners = self.scheduler._hbm_ports.leaked_owners()
        self.assertNotIn(RID + "#prefill_read", owners)
        self.assertIn(RID + "#readplan", owners)
        # §四.1/§四.2：PARTIAL 基 decode credit 计划只覆盖 [0, p)。
        plan = self.runtime.remote_read_credit_plan
        self.assertIsNotNone(plan)
        self.assertEqual(plan["home_instance"], HOME_INSTANCE)
        self.assertEqual(plan["exec_instance"], EXEC_INSTANCE)
        self.assertEqual(plan["read_prefix_layers"], PREFIX_LAYERS)
        self.assertLess(plan["read_prefix_layers"], self.model.layers)

    def test_credit_slices_are_transient_streams_over_prefix(self):
        self.scheduler._on_prefill_drain(RID, 1000)
        blocks = self.scheduler._joint_remote_read_slice(
            self.runtime, 4)
        # auto tile（256 token ≥ 基础 100 token）→ 每 pass 1 tile。
        self.assertEqual(len(blocks), 4)
        for block in blocks:
            # §一.7（前序 KV 管理器卡登记的偏差项，调度器侧补齐）：
            # decode credit 切片同为瞬时读流标记。
            self.assertTrue(block.stream_only)
            self.assertEqual(block.phase, "decode")
            self.assertEqual(block.layer_end, PREFIX_LAYERS)
            self.assertEqual(
                (block.resident_prefix_layers_before,
                 block.resident_prefix_layers_after),
                (PREFIX_LAYERS, PREFIX_LAYERS))

    def test_open_restore_journal_blocks_credit_plan_fail_closed(self):
        # §四.3：suffix restore journal 未关账（恢复链破损）→ drain
        # 在建立 decode credit 计划前 fail-closed。expand_prefill 打桩
        # 跳过 prefill_drain 结算，注入未结算 journal 直达守卫。
        self.scheduler.kv_manager.expand_prefill = lambda **kw: ()
        self.manager._sessions[SID].restore_journal = SimpleNamespace()
        with self.assertRaises(RuntimeError) as caught:
            self.scheduler._on_prefill_drain(RID, 1000)
        message = str(caught.exception)
        self.assertIn("suffix restore", message)
        self.assertIn(SID, message)
        # 守卫先于 credit 计划建立：计划仍为初始值。
        self.assertIsNone(self.runtime.remote_read_credit_plan)


if __name__ == "__main__":
    unittest.main()
