#!/usr/bin/env python3
"""test_pipeline_descriptor_submission.py -- C1 实际发布路径钉测。

链路（全部生产面，不直连私有 _emit_train）：

  真实调度器壳（test_prefill_remote_read_lifecycle 同款：真实
  KVCacheManager/topology/远程读准入）+ **真实 GraphBatchBuilder** →
  `_try_admit_request`（remote-read 准入，真实 lease/pipeline identity
  seeds）→ `_plan_and_emit_trains`（生产列车冻结与发射——prefill read
  window / decode credit stream 的 descriptor 在原发射点登记）→
  **`OnlineSchedulerBase.build_graph_batch`**（组装 + 严格
  `finalize_pipeline_identities` → `batch["pipeline_descriptors"]` 发布
  + kernel_contract）→ 发布批次 JSON。

钉测：
  1. 发布批次为 schema2 完整结构（nodes/parent_edges/watches/
     assignments/kernel_contract/pipeline_descriptors），descriptor wire
     经 `joint.pipeline_descriptor` 原解析器 from_wire 复核；
  2. digest 行的 pipeline_descriptor_count/content 与发布批次一致
     （finalize 先于 digest——发布次序修复的行为钉）；
  3. P→D 同存活 lease 身份连续性：同一 staging allocation 跨 P→D 存活
     时，P 批 prefill_read_window 与 D 批 decode_remote_credit 的
     buffer_domain_id 完全一致（同一 allocation_id）；segment ordinal
     按各自 (request, stage, generation, stage_epoch, rank) 键单调；
     对照面：lease 释放后重新获取 → allocation_id 前移 → 新 domain。

运行：cd sh_test_mesh/workload/llama2_7b_inference &&
      PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -p no:cacheprovider \\
          online/test_pipeline_descriptor_submission.py -q
"""
import json
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

_ONLINE_DIR = os.path.dirname(os.path.abspath(__file__))
_WORKLOAD_DIR = os.path.dirname(_ONLINE_DIR)
for _p in (_ONLINE_DIR, _WORKLOAD_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from joint.prediction_config import (  # noqa: E402
    JointPredictionConfig,
    PredictionParams,
)
from online.test_prefill_remote_read_lifecycle import (  # noqa: E402
    EXEC_INSTANCE,
    HISTORY_TOKENS,
    INPUT_TOKENS,
    PREFIX_LAYERS,
    RID,
    SID,
    TILE_BYTES,
    _force_remote_read,
    _make_manager,
    _make_runtime,
    _model,
    _seed_partial,
)
from face_scheduler import (  # noqa: E402
    kv_cache_shard_bytes_for_tokens,
)
from joint.hbm_port_flow_registry import HbmPortFlowRegistry  # noqa: E402
from joint.joint_config import parse_joint_config  # noqa: E402
from joint.joint_cost_model import (  # noqa: E402
    CausalHorizonEstimator,
    JointHardwareRates,
    LinkFlowRegistry,
    ServiceFactors,
)
from joint.predictor_cost_accounting import (  # noqa: E402
    PredictorCostAccounting,
)
from online.graph_batch_builder import GraphBatchBuilder  # noqa: E402
from online.sh30_online_scheduler import (  # noqa: E402
    _OnlineInstanceState,
    _PoolPortRegistry,
    _TASK_LOAD_CACHE_CAPACITY,
)
import online.sh30_online_scheduler as sh30  # noqa: E402
from online.online_scheduler_base import (  # noqa: E402
    _PROPAGATING_TAIL_SOURCE_ARRIVALS,
    _PROPAGATING_TAIL_SOURCE_DELIVERY,
    _PROPAGATING_TAIL_SOURCE_KV,
    OnlineSchedulerBase,
    PropagatingTailTracker,
)

PUBLISHED_BATCH_DIR = os.environ.get("JOINT_PUBLISHED_BATCH_DIR")

_EVIDENCE_ROOT = os.environ.get(
    "JOINT_ACCEL_EVIDENCE_ROOT",
    "/home/sunhao/joint_accel_execution_20261005.RCuiy5")
_NATIVE_LIBRARY = os.environ.get(
    "JOINT_NATIVE_PROJECTION_LIBRARY",
    os.path.join(
        _EVIDENCE_ROOT,
        "packages/cpp-core/build/congestion_aware/lib/"
        "libjoint_projection.so"))


def _inject_native_library(test):
    """同 ABI 测试注入（§7：测试注入入口只传同一 ABI 的受控 context）：
    裸仓无编译库，D 边界决策的 native 投影经证据根的有效库供给。"""
    if not os.path.isfile(_NATIVE_LIBRARY):
        test.skipTest(
            f"native projection library not found at {_NATIVE_LIBRARY!r}")
    import joint.native_projection as native_module
    from joint.native_projection import NativeProjectionLibrary
    library = NativeProjectionLibrary(_NATIVE_LIBRARY)
    original = native_module.load_production_projection_library
    native_module.load_production_projection_library = lambda: library
    test.addCleanup(
        lambda: setattr(
            native_module, "load_production_projection_library", original))


def _proof_prediction_config() -> JointPredictionConfig:
    return JointPredictionConfig(params=PredictionParams(
        projection_event_budget=4_194_304,
        window_tile_limit=1,
        window_local_error_ppm=10_000,
        window_refine_margin_ppm=20_000,
    ))


def _assemble_production_batch(scheduler, tick: int, seq: int = 0) -> dict:
    """生产组装段（on_decision_batch 尾段）：`build_graph_batch`
    （组装 + 严格 finalize → pipeline_descriptors 发布）。"""
    return OnlineSchedulerBase.build_graph_batch(scheduler)


def _start_delivery(scheduler, tick: int, seq: int) -> None:
    """生产交付开口（on_decision_batch 首段 `_start_batch`）：重建批次
    累加器 + `_begin_graph_batch` → `begin_batch`（每交付恰一次，
    发射前调用）。"""
    scheduler._start_batch({
        "schema_version": 1,
        "delivery_sequence": seq,
        "delivery_epoch": seq,
        "tick": tick,
        "deferred_from_tick": tick,
        "reasons": ["ARRIVAL"],
        "arrivals": [],
        "completed_groups": [],
        "completed_nodes": [],
        "affected_ranks": [],
        "snapshot_handle": {"epoch": seq, "tick": tick},
    })


def _descriptor_identities(batch: dict) -> list:
    """发布批次内 descriptor 的身份面（segment/domain/allocation）。"""
    rows = []
    for wire in batch["pipeline_descriptors"]:
        identity = wire["identity"]
        rows.append({
            "segment_id": identity["segment_id"],
            "buffer_domain_id": identity["buffer_domain_id"],
            "stage": identity["stage"],
            "rank": identity["rank"],
            "staging_allocation_id": identity["staging_allocation_id"],
            "stream_kind": wire["stream_kind"],
        })
    return rows


def _make_scheduler_numeric_pg(model, hardware, topology, manager):
    """生产代表性数字 pg_name 的发布壳（主 agent 裁决 2026-10-06）：
    trace_config 约定 instance i 的 pg_name = str(i+1)，与
    comm_group.json 数字键一致——C++ Workload::parse_comm_group_id_
    strict 只认数字（原行为，不加符号别名）。其余装配与
    test_prefill_remote_read_lifecycle._make_scheduler 同款（准入批
    turn 握手替身，列车/完成/merge 图面全真实）。"""
    scheduler = sh30.Sh30OnlineScheduler.__new__(
        sh30.Sh30OnlineScheduler)
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
        {"JOINT_REMOTE_READ_TILE_BYTES": TILE_BYTES,
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
        _OnlineInstanceState(index=i)
        for i in range(len(topology.instances))]
    scheduler._train_max_iter = 5
    scheduler.runtime_by_request_id = {}
    scheduler.online_log_rows = []
    scheduler.online_log_count = 0
    scheduler.decision_log_sink = None
    # 生产 trace_config 约定：instance i 的 pg_name = str(i+1)（数字串，
    # 与 comm_group.json 数字键 "1"/"2" 一致）。
    graph_config = SimpleNamespace(
        npus_count=4, remote_operand_loads=False,
        trace_granularity="request_aggregated",
        inference_groups=[
            SimpleNamespace(ranks=(0, 1), pg_name="1"),
            SimpleNamespace(ranks=(2, 3), pg_name="2")],
        layers=model.layers, hidden_size=model.hidden_size,
        ffn_size=model.ffn_size, vocab_size=model.vocab_size,
        bytes_per_elem=model.bytes_per_elem,
        num_heads=model.num_heads, mlp_variant=model.mlp_variant,
        request_queue=[SimpleNamespace(
            session_arrival_time_ns=0,
            inter_request_interval_ns=None)])
    scheduler.graph = GraphBatchBuilder(graph_config)
    scheduler.graph.begin_batch()
    scheduler.graph.set_plan_resolver(
        lambda request_id: (
            scheduler.runtime_by_request_id[request_id].plan_dict()))
    scheduler.graph.set_next_plan({RID: None})
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


class PublishedBatchProductionPathTest(unittest.TestCase):
    """真实发布路径：生产发射 → 严格 finalize → schema2 批次。"""

    def setUp(self):
        self.model = _model()
        self.hardware, self.topology, self.manager = _make_manager(
            self.model)
        _seed_partial(self.manager)
        self.scheduler = _make_scheduler_numeric_pg(
            self.model, self.hardware, self.topology, self.manager)
        # build_graph_batch 的基类账面（生产 __init__ 恒设；__new__ 壳
        # 显式补齐——与既有壳装配纪律同款）。
        self.scheduler._joint_prediction_config = (
            _proof_prediction_config())
        self.scheduler._provisional_kv_actions = {}
        self.scheduler.propagating_tail = PropagatingTailTracker()
        self.scheduler.propagating_tail.register(
            _PROPAGATING_TAIL_SOURCE_ARRIVALS,
            lambda: self.scheduler.in_flight, 4)
        self.scheduler.propagating_tail.register(
            _PROPAGATING_TAIL_SOURCE_DELIVERY,
            lambda: self.scheduler._emitted_by_delivery, 8)
        self.scheduler.propagating_tail.register(
            _PROPAGATING_TAIL_SOURCE_KV,
            lambda: self.scheduler._provisional_kv_actions, 8)
        self.scheduler.in_flight = {}
        self.scheduler.digest_rows = []
        self.scheduler.digest_sink = lambda row: (
            self.scheduler.digest_rows.append(row))
        self.scheduler.profile_sink = None
        self.scheduler._profile_batch = {"scanned_entries": 0}
        self.scheduler._batch.setdefault("kv_actions", [])
        self.runtime = _make_runtime()

    def _admit(self, tick: int = 500):
        original_select = _force_remote_read()
        try:
            admitted = self.scheduler._try_admit_request(
                self.runtime, tick)
        finally:
            sh30.select_instance_and_action = original_select
        self.assertTrue(admitted)
        self.scheduler.runtime_by_request_id[RID] = self.runtime
        self.scheduler.runtimes.append(self.runtime)
        self.scheduler._runtime_index[RID] = 0

    def test_p_batch_publishes_prefill_read_window_descriptor(self):
        self._admit()
        _start_delivery(self.scheduler, 600, 0)
        self.scheduler._plan_and_emit_trains(600)
        batch = _assemble_production_batch(self.scheduler, 600, 0)
        self.assertEqual(batch["schema_version"], 2)
        for key in ("batch_id", "source_delivery_sequence", "nodes",
                    "parent_edges", "watches", "assignments", "kv_actions",
                    "future_alarms", "touched_ranks",
                    "pipeline_descriptors", "kernel_contract"):
            self.assertIn(key, batch)
        self.assertGreater(len(batch["nodes"]), 0)
        # 主 agent 裁决（2026-10-06）：collective 节点的 coll.pg_name 必须是
        # 生产代表性数字串（instance i → str(i+1)，与 comm_group.json 数字
        # 键一致）；C++ parse_comm_group_id_strict 只认数字。
        collective_pg = {
            node["coll"]["pg_name"] for node in batch["nodes"]
            if node["type"] == 7}
        self.assertTrue(collective_pg)
        self.assertTrue(
            all(pg.isdigit() for pg in collective_pg),
            f"collective pg_name must be numeric strings, got {collective_pg}")
        identities = _descriptor_identities(batch)
        prefill_rows = [row for row in identities
                        if row["stream_kind"] == "prefill_read_window"]
        self.assertTrue(
            prefill_rows,
            "P batch must publish the prefill_read_window descriptor "
            "through the production emission sequence")
        # 严格 wire 复核：原解析器 from_wire 接受发布形态。
        from joint.pipeline_descriptor import PipelineDescriptor
        for wire in batch["pipeline_descriptors"]:
            parsed = PipelineDescriptor.from_wire(wire)
            self.assertEqual(parsed.identity.stage_wire_label()
                             if hasattr(parsed.identity, "stage_wire_label")
                             else parsed.identity.stage,
                             wire["identity"]["stage"])
        # kernel_contract 九字段（§7 严格合同）。
        contract = batch["kernel_contract"]
        self.assertEqual(
            sorted(contract),
            ["descriptor_version", "model_version",
             "pipeline_protocol_version", "prediction_config_sha256",
             "projection_abi_version", "timer_whitelist_version",
             "window_local_error_ppm", "window_refine_margin_ppm",
             "window_tile_limit"])
        # digest 行与发布批次一致（finalize 先于 digest 的行为钉）。
        self.assertEqual(len(self.scheduler.digest_rows), 1)
        digest = self.scheduler.digest_rows[0]
        self.assertEqual(
            digest["pipeline_descriptor_count"],
            len(batch["pipeline_descriptors"]))
        payload = json.dumps(
            {"nodes": batch["nodes"],
             "parent_edges": batch["parent_edges"],
             "pipeline_descriptors": batch["pipeline_descriptors"],
             "kernel_contract": batch["kernel_contract"]},
            sort_keys=True, separators=(",", ":"))
        import hashlib
        self.assertEqual(
            digest["content_sha256"],
            hashlib.sha256(payload.encode()).hexdigest())

    def test_d_batch_publishes_decode_credit_descriptor_and_pd_continuity(
            self):
        """P→D lease 身份连续性（真实序列）：

        * P 批（prefill_read_window descriptor）发布于生产组装点；
        * D 边界后，decode lease 存活期间跨**多条列车/多个发布批**：
          同一 staging allocation ⇒ 全部 decode descriptor 的
          buffer_domain_id 一致，segment ordinal 按键单调连续
          （第 2 条列车续 ordinal，不重置不回退）；
        * P 与 D 属不同 lease 世代（P lease 在 drain 正常关闭、D 新
          获取）：两域不同且 P 域身份不被 D 获取改写（旧义务跨 D
          保留其域身份）；释放后重取 → allocation 前移。
        """
        _inject_native_library(self)
        self._admit()
        _start_delivery(self.scheduler, 600, 0)
        self.scheduler._plan_and_emit_trains(600)
        p_batch = _assemble_production_batch(self.scheduler, 600, 0)
        p_rows = _descriptor_identities(p_batch)
        p_prefill = [row for row in p_rows
                     if row["stream_kind"] == "prefill_read_window"]
        self.assertTrue(p_prefill)
        p_allocation = p_prefill[0]["staging_allocation_id"]
        p_domains = {row["buffer_domain_id"] for row in p_prefill}

        # ---- 列车核销（drain 标记 watch fire）+ drain 边界 ----
        state = self.scheduler.instances[EXEC_INSTANCE]
        train = state.in_flight_train
        self.assertIsNotNone(train, "first train must be in flight")
        plan = train
        self.scheduler._finalize_completed_trains(
            tuple(plan["drain_members"]), (),
            tuple([plan["train_id"]] if plan["sentinel"] else []), 900)
        self.runtime.queued_chunk_load_ns = 0
        self.scheduler._on_prefill_drain(RID, 1000)

        # ---- D 批 1：首条 decode 列车（credit 流建立） ----
        _start_delivery(self.scheduler, 1100, 1)
        self.scheduler._plan_and_emit_trains(1100)
        d_batch_1 = _assemble_production_batch(self.scheduler, 1100, 1)
        d1_rows = _descriptor_identities(d_batch_1)
        d1_credit = [row for row in d1_rows
                     if row["stream_kind"] == "decode_remote_credit"]
        self.assertTrue(
            d1_credit,
            "D batch must publish the decode_remote_credit descriptor")
        live_lease = self.manager.staging_lease_for_request(RID)
        self.assertIsNotNone(live_lease)
        d_allocation = int(live_lease.allocation_id)
        self.assertNotEqual(
            d_allocation, p_allocation,
            "P lease closed at drain (window consumed); the D boundary "
            "acquires the next allocation generation")

        # ---- D 批 2：同 lease 存活下的第二条列车（身份连续） ----
        train2 = state.in_flight_train
        self.assertIsNotNone(train2, "second train must be in flight")
        plan2 = train2
        self.scheduler._finalize_completed_trains(
            tuple(plan2["drain_members"]), (),
            tuple([plan2["train_id"]] if plan2["sentinel"] else []), 1400)
        _start_delivery(self.scheduler, 1500, 2)
        self.scheduler._plan_and_emit_trains(1500)
        d_batch_2 = _assemble_production_batch(self.scheduler, 1500, 2)

        # 同存活 lease：全部 decode descriptor 同 allocation 同域。
        for batch in (d_batch_1, d_batch_2):
            for row in _descriptor_identities(batch):
                if row["stream_kind"] != "decode_remote_credit":
                    continue
                self.assertEqual(row["staging_allocation_id"],
                                 d_allocation)
                self.assertIn(
                    row["buffer_domain_id"],
                    {r["buffer_domain_id"]
                     for r in d1_credit})
        # segment ordinal 连续性：decode 键跨批单调（不重置）。
        decode_ordinals = []
        for batch in (d_batch_1, d_batch_2):
            for row in _descriptor_identities(batch):
                if row["stream_kind"] == "decode_remote_credit":
                    decode_ordinals.append(
                        (row["rank"],
                         int(row["segment_id"].rsplit("/", 1)[1])))
        by_rank = {}
        for rank, ordinal in decode_ordinals:
            by_rank.setdefault(rank, []).append(ordinal)
        for rank, ordinals in by_rank.items():
            self.assertEqual(ordinals, sorted(ordinals),
                             f"decode ordinals for rank {rank} must stay "
                             "monotonic across batches under one lease")
        # P 域身份不被 D 获取改写（旧义务跨 D 保留其域）。
        refreshed_p = [
            row for row in _descriptor_identities(p_batch)
            if row["stream_kind"] == "prefill_read_window"]
        self.assertEqual(
            {row["buffer_domain_id"] for row in refreshed_p}, p_domains)
        self.assertTrue(p_domains.isdisjoint({
            row["buffer_domain_id"] for row in d1_credit}))

        # ---- 对照面：lease 释放后重新获取 → allocation 前移 ----
        self.manager.release_staging_lease(RID)
        self.assertIsNone(self.manager.staging_lease_for_request(RID))
        self.assertGreater(
            self.manager.peek_next_staging_allocation_id(), d_allocation)

    def test_publishes_schema2_batch_artifacts_when_dir_env_set(self):
        """落盘真实发布批次（P/D 两批）供 C3 真实 K1 旧新差分。"""
        if not PUBLISHED_BATCH_DIR:
            self.skipTest("JOINT_PUBLISHED_BATCH_DIR not set")
        _inject_native_library(self)
        self._admit()
        _start_delivery(self.scheduler, 600, 0)
        self.scheduler._plan_and_emit_trains(600)
        p_batch = _assemble_production_batch(self.scheduler, 600, 0)
        state = self.scheduler.instances[EXEC_INSTANCE]
        plan = state.in_flight_train
        self.scheduler._finalize_completed_trains(
            tuple(plan["drain_members"]), (),
            tuple([plan["train_id"]] if plan["sentinel"] else []), 900)
        self.runtime.queued_chunk_load_ns = 0
        self.scheduler._on_prefill_drain(RID, 1000)
        _start_delivery(self.scheduler, 1100, 1)
        self.scheduler._plan_and_emit_trains(1100)
        d_batch = _assemble_production_batch(self.scheduler, 1100, 1)
        out = Path(PUBLISHED_BATCH_DIR)
        out.mkdir(parents=True, exist_ok=True)
        p_path = out / "published_batch_P_schema2.json"
        d_path = out / "published_batch_D_schema2.json"
        p_path.write_text(json.dumps(p_batch, indent=1, sort_keys=True),
                          encoding="utf-8")
        d_path.write_text(json.dumps(d_batch, indent=1, sort_keys=True),
                          encoding="utf-8")
        inputs_path = out / "builder_inputs.json"
        inputs_path.write_text(json.dumps({
            "p_batch": {
                "nodes": len(p_batch["nodes"]),
                "edges": len(p_batch["parent_edges"]),
                "descriptors": [
                    row["segment_id"] for row in
                    _descriptor_identities(p_batch)],
            },
            "d_batch": {
                "nodes": len(d_batch["nodes"]),
                "edges": len(d_batch["parent_edges"]),
                "descriptors": [
                    row["segment_id"] for row in
                    _descriptor_identities(d_batch)],
            },
            "fixture": {
                "model_layers": self.model.layers,
                "history_tokens": HISTORY_TOKENS,
                "input_tokens": INPUT_TOKENS,
                "prefix_layers": PREFIX_LAYERS,
                "session_id": SID,
                "request_id": RID,
                "exec_instance": EXEC_INSTANCE,
            },
        }, indent=1, sort_keys=True), encoding="utf-8")
        self.assertTrue(d_batch["pipeline_descriptors"])


if __name__ == "__main__":
    unittest.main()
