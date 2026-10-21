#!/usr/bin/env python3
"""test_acceptance_projection.py -- 有界真实 P 边界重建证明（§19.1 L3）。

链路：真实在线对象（Sh30OnlineScheduler ``__new__`` 装配 + 真实
KVCacheManager/topology/hardware/model + 生产 ``_joint_cost_model`` +
原 ``select_instance_and_action``）→ 只读 exporter 导出 boundary
JSON → 落盘 → **独立进程** hydrator 按原构造器重建 →
``evaluate_boundary`` → 与原进程内选择结果对照：

  * 同候选面（同序全菜单 candidate_id 列表）；
  * 同适用性（applicable/inapplicable 分面一致）；
  * 同选择（selected_id == 原进程 chosen）；
  * 成本一致（逐候选 results 行：cost_ns/service_ns/settle_ns）；
  * projection_input / work_item typed wire 一致。

native 库 = 有效新构建的 libjoint_projection.so（环境变量
``JOINT_NATIVE_PROJECTION_LIBRARY`` 或证据根默认路径）；证据落盘目录
= ``JOINT_ACCEPTANCE_PROOF_DIR``（缺省 tempdir，不写仓库）。

运行：cd sh_test_mesh/workload/llama2_7b_inference &&
      PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -p no:cacheprovider \\
          joint/test_acceptance_projection.py -q
"""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
for _path in (_HERE, _PARENT):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from face_scheduler import (  # noqa: E402
    FaceHardware,
    FaceInstanceSpec,
    FaceModel,
    KVCacheManager,
    build_instances,
    kv_cache_shard_bytes_for_tokens,
)
from joint.hbm_port_flow_registry import (  # noqa: E402
    HbmPortFlowRegistry,
)
from joint.joint_config import parse_joint_config  # noqa: E402
from joint.joint_cost_model import (  # noqa: E402
    CausalHorizonEstimator,
    JointHardwareRates,
    LinkFlowRegistry,
    RequestView,
    ServiceFactors,
)
from joint.joint_scheduler import (  # noqa: E402
    _candidate_id,
    select_instance_and_action,
)
from joint.prediction_config import (  # noqa: E402
    JointPredictionConfig,
    PredictionParams,
)
from online.sh30_online_scheduler import (  # noqa: E402
    _OnlineInstanceState,
    _PoolPortRegistry,
    _TASK_LOAD_CACHE_CAPACITY,
    Sh30OnlineScheduler,
)
from joint.predictor_cost_accounting import (  # noqa: E402
    PredictorCostAccounting,
)

_EVIDENCE_ROOT = os.environ.get(
    "JOINT_ACCEL_EVIDENCE_ROOT",
    "/home/sunhao/joint_accel_execution_20261005.RCuiy5")
_DEFAULT_LIBRARY = os.path.join(
    _EVIDENCE_ROOT,
    "packages/cpp-core/build/congestion_aware/lib/libjoint_projection.so")
NATIVE_LIBRARY = os.environ.get(
    "JOINT_NATIVE_PROJECTION_LIBRARY", _DEFAULT_LIBRARY)
PROOF_DIR = os.environ.get("JOINT_ACCEPTANCE_PROOF_DIR")

SESSION_ID = "s_proof"
REQUEST_ID = "r_proof"
HISTORY_TOKENS = 96
INPUT_TOKENS = 16


def _make_scheduler() -> Sh30OnlineScheduler:
    """装配真实生产面（与 test_joint_decision_schema 同款 __new__ 壳，
    叠加真实 KV 会话历史/背景流/遥测，使候选面与注册表非平凡）。"""
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
        pool_latency_ns=0,
    )
    scheduler._instance_edge_ports = {}
    scheduler.joint_config = parse_joint_config(env={})
    scheduler._joint_mode = scheduler.joint_config.scheduler_mode
    scheduler._prefill_task_cache = {}
    scheduler._decode_task_load_cache = {}
    scheduler._task_load_cache_capacity = _TASK_LOAD_CACHE_CAPACITY
    scheduler._quota_tracker = None
    scheduler._link_telemetry_rates = {}
    scheduler._link_telemetry_flow_counts = {}
    scheduler.predictor_cost = PredictorCostAccounting()
    scheduler._telemetry_last_tick_ns = 0
    scheduler._telemetry_last_window_bounds_ns = None
    scheduler._snapshot_verify = False
    scheduler._kv_ledger_epoch = 0
    scheduler._ready_frontier = set()
    scheduler.online_log_rows = []
    scheduler.online_log_count = 0
    scheduler.decision_log_sink = None
    scheduler._joint_prediction_config = _proof_prediction_config()
    # 真实 KV 历史：turn-0 建立在实例 1，完成后进入下一轮基础。
    scheduler.kv_manager.prepare_prefill(
        session_id=SESSION_ID, target_instance_index=1,
        history_tokens=0, trigger_request_id="seed")
    scheduler.kv_manager.expand_prefill(
        session_id=SESSION_ID, instance_index=1,
        context_tokens=HISTORY_TOKENS, trigger_request_id="seed")
    scheduler.kv_manager.mark_complete(SESSION_ID, 64)
    # 背景在途流（链路 0->2 两流、池端口 1 流、HBM 端口流 1 条）：
    # 注册表面非平凡，导出/重建两侧同源。
    scheduler._joint_flows.register_path((0, 2), owner="bg#1")
    scheduler._joint_flows.register_path((0, 2), owner="bg#2")
    scheduler._pool_ports.register(1, owner="bg#pool")
    scheduler._hbm_ports.register(2, owner="bg#hbm")
    scheduler._hbm_ports.attach_active_decode_provider(lambda rank: 0)
    return scheduler


def _proof_prediction_config() -> JointPredictionConfig:
    """精确冒烟配置（§7：精确单 timer 路径，预算 4194304）。"""
    return JointPredictionConfig(params=PredictionParams(
        projection_event_budget=4_194_304,
    ))


def _build_views(scheduler):
    """生产配方构造 session/request 视图（_try_admit_request 同款）。"""
    session_view = scheduler._joint_session_view(SESSION_ID)
    _conditional = scheduler._joint_conditional_estimate(
        _RuntimeProbe(SESSION_ID))
    estimated_decode = int(_conditional.remaining_forward_steps)
    horizon_source = "conditional:" + _conditional.source
    input_kv_shards = kv_cache_shard_bytes_for_tokens(
        scheduler.model, INPUT_TOKENS, scheduler.kv_manager.tp_degree)
    request_view = RequestView(
        request_id=REQUEST_ID,
        session_id=SESSION_ID,
        input_tokens=INPUT_TOKENS,
        history_tokens_before=session_view.history_tokens,
        estimated_decode_tokens=estimated_decode,
        horizon_source=horizon_source,
        input_kv_bytes_by_tp_rank=tuple(input_kv_shards),
    )
    return session_view, request_view


class _RuntimeProbe:
    """``_joint_conditional_estimate`` 的最小 runtime 面（准入时刻
    decode_tokens_consumed=0、无已揭示类别）。"""

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self.decode_tokens_consumed = 0
        self.joint_revealed_category = None


def _in_process_decision(scheduler):
    """真实生产 P 决策（原选择器 + 生产 cost model + 显式 native 库）。"""
    from joint.native_projection import NativeProjectionLibrary
    session_view, request_view = _build_views(scheduler)
    cost_model = scheduler._joint_cost_model(0)
    cost_model.native_projection_library = NativeProjectionLibrary(
        NATIVE_LIBRARY)
    record = select_instance_and_action(
        mode=scheduler._joint_mode,
        cost_model=cost_model,
        session=session_view,
        request=request_view)
    return record, cost_model, session_view, request_view


def _reference_snapshot(record) -> dict:
    """原进程内选择结果的对照快照（results 行与生产导出器同形状）。"""
    def _evaluation_row(candidate):
        breakdown = candidate.breakdown
        return {
            "candidate_id": _candidate_id(candidate),
            "status": ("ok" if candidate.applicable
                       and candidate.cost_ns is not None
                       else "inapplicable"),
            "applicable": bool(candidate.applicable),
            "inapplicable_reason": candidate.inapplicable_reason,
            "cost_ns": candidate.cost_ns,
            "service_ns": (None if breakdown is None
                           else breakdown.predicted_service_done),
            "settle_ns": (None if breakdown is None
                          else breakdown.predicted_merge_done),
            "native_kernel": (
                {} if breakdown is None
                else dict(breakdown.projection_stats.get(
                    "native_kernel", {}))),
        }
    return {
        "chosen_id": _candidate_id(record.chosen),
        "face": [
            {
                "candidate_id": _candidate_id(candidate),
                "applicable": bool(
                    candidate.applicable
                    and candidate.cost_ns is not None),
                "inapplicable_reason": candidate.inapplicable_reason,
                "cost_ns": candidate.cost_ns,
            }
            for candidate in record.candidates],
        "results": [_evaluation_row(candidate)
                    for candidate in record.candidates],
        "mode": record.mode,
    }


_EVALUATE_DRIVER = r'''
import json, os, sys
workload = sys.argv[1]
boundary_path, fixture_path, out_path, library = sys.argv[2:6]
params_path = sys.argv[6]
sys.path.insert(0, workload)
from joint.acceptance_projection import evaluate_boundary
from joint.prediction_config import (
    JointPredictionConfig, PredictionParams)
boundary = json.load(open(boundary_path, encoding="utf-8"))
fixture_state = json.load(open(fixture_path, encoding="utf-8"))
params = json.load(open(params_path, encoding="utf-8"))
config = JointPredictionConfig(params=PredictionParams(**params))
dto = evaluate_boundary(
    boundary, config, library_path=library, fixture_state=fixture_state)
json.dump(dto, open(out_path, "w", encoding="utf-8"), sort_keys=True)
print(dto.get("status"))
'''


@unittest.skipIf(
    not os.path.isfile(NATIVE_LIBRARY),
    f"native projection library not found at {NATIVE_LIBRARY!r}")
class BoundedRealPBoundaryProofTest(unittest.TestCase):
    """有界真实 P 边界：导出 → 落盘 → 独立进程重建 → 对照。"""

    def test_boundary_export_hydrate_evaluation_matches_in_process(self):
        import joint.acceptance_projection as ap

        scheduler = _make_scheduler()
        record, cost_model, session_view, request_view = (
            _in_process_decision(scheduler))
        self.assertGreaterEqual(
            sum(1 for row in record.candidates
                if row.applicable and row.cost_ns is not None),
            2,
            "bounded proof expects a non-trivial priced candidate face")

        # ---- 只读导出（同一决策时点状态） ----
        metadata = {
            "context_tokens": int(session_view.history_tokens),
            "horizon_tokens": int(request_view.estimated_decode_tokens),
            "forecast_generator": ap.export_forecast_generator(
                context_tokens=int(session_view.history_tokens),
                horizon_tokens=int(
                    request_view.estimated_decode_tokens)),
        }
        planner_state = ap.export_planner_state_P(
            scheduler, cost_model,
            session=session_view,
            request=request_view,
            prep_frontier_by_instance=None,
            context_tokens=metadata["context_tokens"],
            horizon_tokens=metadata["horizon_tokens"])
        runtime_asset = ap.export_runtime_asset(scheduler)
        boundary = ap.export_boundary(
            scheduler, cost_model,
            request_id=REQUEST_ID,
            stage="P",
            boundary_ordinal=0,
            tick_ns=0,
            fixed_instance_id=None,
            fixed_action=None,
            focus_session_id=SESSION_ID,
            planner_state=planner_state,
            causal_metadata=metadata)
        fixture_state = {
            "python_runtime_asset_id": "inline:planner_state_runtime",
            "python_runtime_asset": runtime_asset,
            "assets": {},
        }

        with tempfile.TemporaryDirectory(
                prefix="joint-acceptance-proof-") as tmp:
            proof_dir = Path(PROOF_DIR) if PROOF_DIR else Path(tmp)
            proof_dir.mkdir(parents=True, exist_ok=True)
            boundary_path = proof_dir / "boundary.json"
            fixture_path = proof_dir / "fixture_state.json"
            params_path = proof_dir / "prediction_params.json"
            reference_path = proof_dir / "in_process_reference.json"
            dto_path = proof_dir / "evaluated_dto.json"
            report_path = proof_dir / "comparison_report.json"
            driver_path = proof_dir / "evaluate_driver.py"
            boundary_path.write_text(
                json.dumps(boundary, sort_keys=True, indent=1),
                encoding="utf-8")
            fixture_path.write_text(
                json.dumps(fixture_state, sort_keys=True, indent=1),
                encoding="utf-8")
            params_path.write_text(json.dumps(
                asdict(record and _proof_prediction_config().params)),
                encoding="utf-8")
            reference = _reference_snapshot(record)
            # 对照用 typed wire（同进程内提取；与 DTO 侧逐键比较）。
            from joint.native_projection import (
                _work_wire, projection_input_to_wire)
            reference["work_item_wires"] = {
                _candidate_id(candidate): _work_wire(
                    getattr(getattr(candidate, "projection_graph", None),
                            "work", None))
                for candidate in record.candidates
                if candidate.applicable and candidate.cost_ns is not None}
            chosen_world = getattr(record.chosen, "projection_world", None)
            chosen_graph = getattr(record.chosen, "projection_graph", None)
            reference["projection_input_wire"] = projection_input_to_wire(
                chosen_world.projection_input(chosen_graph))
            reference_path.write_text(
                json.dumps(reference, sort_keys=True, indent=1),
                encoding="utf-8")
            driver_path.write_text(_EVALUATE_DRIVER, encoding="utf-8")

            env = dict(os.environ)
            env["PYTHONDONTWRITEBYTECODE"] = "1"
            completed = subprocess.run(
                [sys.executable, str(driver_path), _PARENT,
                 str(boundary_path), str(fixture_path), str(dto_path),
                 NATIVE_LIBRARY, str(params_path)],
                capture_output=True, text=True, env=env, timeout=600)
            self.assertEqual(
                completed.returncode, 0,
                f"evaluate driver failed:\n{completed.stderr[-4000:]}")
            dto = json.loads(dto_path.read_text(encoding="utf-8"))

            report = _compare(reference, dto)
            report_path.write_text(
                json.dumps(report, sort_keys=True, indent=1),
                encoding="utf-8")
            if PROOF_DIR:
                report["artifacts"] = {
                    "boundary": str(boundary_path),
                    "fixture_state": str(fixture_path),
                    "in_process_reference": str(reference_path),
                    "evaluated_dto": str(dto_path),
                    "comparison_report": str(report_path),
                    "native_library": NATIVE_LIBRARY,
                }
            failed = {key: value for key, value in report.items()
                      if value is False}
            self.assertFalse(
                failed, f"bounded P boundary comparison failed: {failed}")

    def test_pass_dto_contract_shape(self):
        """PASS DTO 字面合同（§19.1 字段集 + 互斥 + forecast 三键）。"""
        import joint.acceptance_projection as ap
        scheduler = _make_scheduler()
        record, cost_model, session_view, request_view = (
            _in_process_decision(scheduler))
        metadata = {
            "context_tokens": int(session_view.history_tokens),
            "horizon_tokens": int(request_view.estimated_decode_tokens),
            "forecast_generator": ap.export_forecast_generator(
                context_tokens=int(session_view.history_tokens),
                horizon_tokens=int(
                    request_view.estimated_decode_tokens)),
        }
        planner_state = ap.export_planner_state_P(
            scheduler, cost_model, session=session_view,
            request=request_view, prep_frontier_by_instance=None,
            context_tokens=metadata["context_tokens"],
            horizon_tokens=metadata["horizon_tokens"])
        boundary = ap.export_boundary(
            scheduler, cost_model, request_id=REQUEST_ID, stage="P",
            boundary_ordinal=0, tick_ns=0, fixed_instance_id=None,
            fixed_action=None, focus_session_id=SESSION_ID,
            planner_state=planner_state, causal_metadata=metadata)
        fixture_state = {
            "python_runtime_asset_id": "inline:planner_state_runtime",
            "python_runtime_asset": ap.export_runtime_asset(scheduler),
            "assets": {},
        }
        dto = ap.evaluate_boundary(
            boundary, _proof_prediction_config(),
            library_path=NATIVE_LIBRARY, fixture_state=fixture_state)
        self.assertEqual(dto["status"], "PASS", dto.get("reason"))
        self.assertEqual(
            dto["version"], "projection-evaluation-v3-20261016")
        for key in ("request_id", "stage", "boundary_ordinal", "tick_ns",
                    "horizon_tokens", "projection_input", "candidates",
                    "inapplicable_candidates", "forecast_generator",
                    "results", "selected_id",
                    "library_sha256", "prediction_config_sha256"):
            self.assertIn(key, dto)
        generator = dto["forecast_generator"]
        self.assertEqual(
            sorted(generator),
            ["causal_inputs", "identity", "semantic_sha256"])
        self.assertEqual(
            sorted(generator["causal_inputs"]),
            ["context_tokens", "generator_inputs", "horizon_tokens"])
        self.assertEqual(
            generator["identity"],
            "no-exogenous-generator/v9-tie-roundrobin-20261020")
        self.assertEqual(generator["causal_inputs"]["generator_inputs"], {})
        self.assertIsInstance(dto["selected_id"], str)
        for row in dto["candidates"]:
            self.assertEqual(
                sorted(row), ["action", "candidate_id", "instance_id",
                              "work_item"])
        for row in dto["inapplicable_candidates"]:
            self.assertEqual(
                sorted(row), ["action", "candidate_id", "instance_id",
                              "reason"])
            self.assertIsInstance(row["reason"], str)
        for row in dto["results"]:
            self.assertEqual(
                sorted(row),
                ["candidate_id", "quality", "result", "service_ns",
                 "settle_ns", "status"])
        # JSON round-trip（可落盘性）。
        self.assertEqual(json.loads(json.dumps(dto)), dto)

    def test_failure_dto_is_minimal_on_missing_fixture(self):
        import joint.acceptance_projection as ap
        scheduler = _make_scheduler()
        record, cost_model, session_view, request_view = (
            _in_process_decision(scheduler))
        metadata = {
            "context_tokens": 0,
            "horizon_tokens": 1,
            "forecast_generator": ap.export_forecast_generator(
                context_tokens=0, horizon_tokens=1),
        }
        planner_state = ap.export_planner_state_P(
            scheduler, cost_model, session=session_view,
            request=request_view, prep_frontier_by_instance=None,
            context_tokens=0, horizon_tokens=1)
        boundary = ap.export_boundary(
            scheduler, cost_model, request_id=REQUEST_ID, stage="P",
            boundary_ordinal=0, tick_ns=0, fixed_instance_id=None,
            fixed_action=None, focus_session_id=SESSION_ID,
            planner_state=planner_state, causal_metadata=metadata)
        dto = ap.evaluate_boundary(
            boundary, _proof_prediction_config(),
            library_path=NATIVE_LIBRARY, fixture_state=None)
        self.assertEqual(dto["status"], "INCOMPLETE")
        self.assertEqual(
            dto["version"], "projection-evaluation-v3-20261016")
        self.assertIsInstance(dto["reason"], str)
        self.assertNotIn("candidates", dto)
        self.assertNotIn("results", dto)
        self.assertEqual(dto["request_id"], REQUEST_ID)


def _compare(reference: dict, dto: dict) -> dict:
    """对照报告（同候选面/同适用性/同选择/同 results 行/同 typed wire）。"""
    from joint.joint_cost_model import ACTION_ORDER

    ref_face = reference["face"]
    dto_applicable = dto["candidates"]
    dto_inapplicable = dto["inapplicable_candidates"]
    dto_face_ids = ([row["candidate_id"] for row in dto_applicable]
                    + [row["candidate_id"] for row in dto_inapplicable])
    ref_face_ids = [row["candidate_id"] for row in ref_face]
    ref_applicable_ids = [row["candidate_id"] for row in ref_face
                          if row["applicable"]]
    ref_inapplicable_ids = [row["candidate_id"] for row in ref_face
                            if not row["applicable"]]

    def menu_rank(candidate_id: str):
        instance, _, action = candidate_id.partition(":")
        return (int(instance), ACTION_ORDER.index(action))

    merged_ids = sorted(dto_face_ids, key=menu_rank)
    results_by_id = {
        row["candidate_id"]: row for row in dto["results"]}
    ref_results = {
        row["candidate_id"]: row
        for row in reference["results"]}
    rows_match = set(results_by_id) == set(ref_results)
    if rows_match:
        for candidate_id, row in ref_results.items():
            dto_row = results_by_id[candidate_id]
            if (dto_row["status"] != row["status"]
                    or dto_row["service_ns"] != row["service_ns"]
                    or dto_row["settle_ns"] != row["settle_ns"]
                    or dto_row["result"] != row):
                rows_match = False
                break
    work_wires_match = None
    if reference.get("work_item_wires") is not None:
        dto_wires = {
            row["candidate_id"]: row["work_item"]
            for row in dto_applicable}
        work_wires_match = dto_wires == {
            key: value for key, value in
            reference["work_item_wires"].items()}
    projection_input_match = None
    if reference.get("projection_input_wire") is not None:
        projection_input_match = (
            dto["projection_input"] == reference["projection_input_wire"])
    return {
        "same_candidate_face_order": merged_ids == ref_face_ids,
        "same_candidate_face_ids": (
            set(dto_face_ids) == set(ref_face_ids)
            and len(dto_face_ids) == len(ref_face_ids)),
        "same_applicability_split": (
            [row["candidate_id"] for row in dto_applicable]
            == ref_applicable_ids
            and [row["candidate_id"] for row in dto_inapplicable]
            == ref_inapplicable_ids),
        "same_selected": dto["selected_id"] == reference["chosen_id"],
        "same_results_rows": rows_match,
        "same_work_item_wires": work_wires_match,
        "same_projection_input_wire": projection_input_match,
        "dto_status": dto.get("status"),
        "dto_reason": dto.get("reason"),
        "reference_chosen": reference["chosen_id"],
        "dto_selected": dto.get("selected_id"),
    }


class ForecastEngineStableProjectionTest(unittest.TestCase):
    """registries.forecast_history[*].engine 的稳定投影（双副本确定性）。

    str(engine) 含进程地址 → 两个 fresh-process 副本的 canonical 结果
    在该字段永不相等，capture 双副本确定性比较必然失败。无结构化导出
    面的 engine 必须序列化为固定 token '<engine-object>'。
    """

    def test_engine_without_structured_surface_exports_stable_token(self):
        import joint.acceptance_projection as ap

        class _OpaqueEngine:
            pass  # 故意无 as_dict / dataclass 面

        class _Scheduler:
            def _joint_forecast_engine_or_default(self):
                return _OpaqueEngine()

        history_a = ap._export_forecast_history(_Scheduler())
        history_b = ap._export_forecast_history(_Scheduler())
        self.assertEqual(history_a, [{"engine": "<engine-object>"}])
        self.assertEqual(history_a, history_b)
        self.assertNotIn("0x", json.dumps(history_a))

    def test_engine_with_structured_surface_exports_facts(self):
        import joint.acceptance_projection as ap

        class _DictEngine:
            def as_dict(self):
                return {"mode": "none"}

        class _Scheduler:
            def _joint_forecast_engine_or_default(self):
                return _DictEngine()

        history = ap._export_forecast_history(_Scheduler())
        self.assertEqual(history, [{"engine": {"mode": "none"}}])

    def test_missing_engine_exports_empty_array(self):
        import joint.acceptance_projection as ap

        class _Scheduler:
            def _joint_forecast_engine_or_default(self):
                return None

        self.assertEqual(ap._export_forecast_history(_Scheduler()), [])


if __name__ == "__main__":
    unittest.main()


_C4_FRAMES_PATH = os.path.join(
    _EVIDENCE_ROOT, "manifests/c4-v2-runtime-microframes-20261006_009.json")


def _stage_b_node_only_v3_frames(frozen_frames):
    """Build a legal v3 Stage B fixture from the frozen ordinary-node tape.

    The source evidence predates sparse protocol v3 and also contains descriptor
    lifecycle events.  This test covers Stage B's ordinary WorkLedger replay,
    so retain its node events and ticks while assigning the node-only stream a
    contiguous logical sequence.  Pipeline state is intentionally absent and
    represented by empty v3 arrays; no legacy envelope or pipeline rows are
    passed to the production reader.
    """
    epoch = "stage-b-c4-node-only-v3"
    frames = []
    watermark = 0
    state_version = 0
    for delivery, source in enumerate(frozen_frames):
        node_rows = []
        for source_event in source.get("node_transitions", ()):
            event = dict(source_event)
            watermark += 1
            event["logical_seq"] = watermark
            node_rows.append(event)
        event_count = len(node_rows)
        base_version = state_version
        if event_count:
            state_version += 1
        frames.append({
            "producer": "cpp_online",
            "schema_version": 3,
            "protocol_version": 3,
            "bridge_protocol": "bridge-v3-20261009",
            "work_ledger_schema": "workledger-v3-20261009",
            "epoch": epoch,
            "delivery_sequence": delivery,
            "base_watermark": watermark - event_count,
            "logical_seq_start": watermark - event_count + 1,
            "logical_seq_end": watermark,
            "logical_event_count": event_count,
            "replay_token": f"cpp_online/{epoch}/{delivery}",
            "descriptor_version": 1,
            "base_state_version": base_version,
            "result_state_version": state_version,
            "snapshot_kind": "full" if delivery == 0 else "delta",
            "tick": int(source["tick"]),
            "descriptor_commits": [],
            "node_transitions": node_rows,
            "pipeline_transition_runs": [],
            "cursor_patches": [],
            "active_resources": [],
        })
    return frames


def _empty_v4_retained_frame():
    return {
        "producer": "cpp_online",
        "schema_version": 4,
        "protocol_version": 4,
        "bridge_protocol": "bridge-v4-20261011",
        "work_ledger_schema": "workledger-v4-20261011",
        "epoch": "retained-v4-test",
        "delivery_sequence": 0,
        "base_watermark": 0,
        "logical_seq_start": 1,
        "logical_seq_end": 0,
        "logical_event_count": 0,
        "replay_token": "cpp_online/retained-v4-test/0",
        "descriptor_version": 1,
        "base_state_version": 0,
        "result_state_version": 0,
        "snapshot_kind": "full",
        "tick": 5,
        "descriptor_commits": [],
        "node_transitions": [],
        "pipeline_transition_runs": [],
        "cursor_patches": [],
        "active_resources": [],
    }


def _driver_validated_state(scheduler, state_dir):
    """构造驱动 validate_fixture_state 可接受的冷态（精确键集、无额外
    键；assets 为列表、runtime asset 落盘相对路径 + SHA256）。"""
    import joint.acceptance_projection as ap

    session_view, request_view = _build_views(scheduler)
    asset = ap.export_runtime_asset(scheduler)
    asset_path = Path(state_dir) / "runtime_asset.json"
    asset_path.write_text(
        json.dumps(asset, sort_keys=True, indent=1), encoding="utf-8")
    import hashlib
    asset_sha = hashlib.sha256(asset_path.read_bytes()).hexdigest()
    hardware = scheduler.topology.hardware
    rank_count = hardware.mesh_rows * hardware.mesh_cols
    initial_state = {
        "tick_ns": 0, "next_flow_id": 0, "next_tag": 1,
        "ranks": [
            {"rank": rank, "next_store_id": 1}
            for rank in range(rank_count)],
        "components": [
            {"identity": f"rank{rank}/noc", "version": 0, "state": {}}
            for rank in range(rank_count)],
        # loaded logical cold initial state（动态组全空、逻辑面可装载）。
        "nodes": [], "ready": [], "deferred": [], "events": [],
        "flows": {}, "tails": [], "hbm_jobs": [], "pool_jobs": [],
        "watch": {}, "descriptors": {}, "cursors": {}, "buffers": {},
        "obligations": {}, "hardware_inflight": {},
        "directory": {"present": True, "version": 0, "entries": [],
                      "inflight_copies": []},
        "capacity": {"present": True, "ledgers_by_instance": {},
                     "summary": {"present": True, "instances": {}}},
        "history": {"horizon_estimator": None,
                    "conditional_length": None},
    }
    return {
        "version": "execution-fixture-v1-20261006",
        "kind": "cold_initial",
        "case_id": "stage-b-bounded",
        "cluster_id": "M01/seed000",
        "scenario_id": "M01",
        "seed": 0,
        "request_id": "r_stage_b",
        "fixed_instance_id": "ins0",
        "fixed_action": "stay",
        "causal_input": {
            "focus_stage": "D",
            "context_tokens": 512,
            "horizon_tokens": 8,
            "competition": "low",
            "source_count": 1,
        },
        "assets": [
            {"asset_id": "runtime", "path": "runtime_asset.json",
             "sha256": asset_sha},
        ],
        "hardware": {
            "rank_count": rank_count,
            "system_asset_id": "runtime",
            "network_asset_id": "runtime",
            "remote_memory_asset_id": "runtime",
            "die_hardware_asset_id": None,
        },
        "initial_state": initial_state,
        "python_runtime_asset_id": "runtime",
    }


def _raw_result(frames, *, p_instance="ins0", p_action="stay",
                d_instance="ins1", d_action="stay", ordinals=(0, 0),
                frame_ordinal_override=None, pending=True):
    """最小 raw result：数组审计面包两条 raw_v2_frame 外层记录，内部
    为 v3 work-ledger 帧（P/D 边界各一，同 boundary_ordinal=0——按提交
    位置配对）。"""
    boundary_records = [
        {"request_id": "r_stage_b", "stage": "P",
         "boundary_ordinal": ordinals[0], "tick_ns": frames[0]["tick"],
         "fixed_instance_id": p_instance, "fixed_action": p_action,
         "causal_prestate": None, "planner_state": None,
         "causal_metadata": {"context_tokens": 512, "horizon_tokens": 8}},
        {"request_id": "r_stage_b", "stage": "D",
         "boundary_ordinal": ordinals[1], "tick_ns": frames[1]["tick"],
         "fixed_instance_id": d_instance, "fixed_action": d_action,
         "causal_prestate": None, "planner_state": None,
         "causal_metadata": {"context_tokens": 512, "horizon_tokens": 8}},
    ]
    audit_records = []
    for position, frame in enumerate(frames[:2]):
        ordinal = ordinals[position]
        if frame_ordinal_override is not None and position == 1:
            ordinal = frame_ordinal_override
        audit_records.append({
            "kind": "raw_v2_frame",
            "boundary_ordinal": ordinal,
            "frame": frame,
            "runtime_snapshot": {
                "tick_ns": frame["tick"], "nodes": [], "hbm": [],
                "pool": [], "noc": []},
            "producer_build": {
                "binary_sha256": "a" * 64, "kernel_contract": {}},
        })
    result = {
        "version": "execution-result-v1-20261006",
        "case_id": "stage-b-bounded",
        "cluster_id": "M01/seed000",
        "scenario_id": "M01",
        "seed": 0,
        "request_id": "r_stage_b",
        "fixed_instance_id": d_instance,
        "fixed_action": d_action,
        "kernel_contract": {
            "projection_abi_version": 1, "descriptor_version": 1,
            "pipeline_protocol_version": 1,
            "model_version": "v9-tie-roundrobin-20261020",
            "timer_whitelist_version": "timer-whitelist-v1-20261005",
            "prediction_config_sha256": "b" * 64},
        "status": "PASS",
        "reason": None,
        "completed": True,
        "initial_certificate": {},
        "terminal_certificate": {},
        "measurements": [
            {"stage": "P", "fixed_instance_id": p_instance,
             "fixed_action": p_action, "boundary_ordinal": ordinals[0],
             "start_tick_ns": 0, "service_tick_ns": 1, "settle_tick_ns": 2,
             "logical_event_count": 1, "native_physical_steps": 1,
             "charged_totals": {}},
            {"stage": "D", "fixed_instance_id": d_instance,
             "fixed_action": d_action, "boundary_ordinal": ordinals[1],
             "start_tick_ns": 0, "service_tick_ns": 1, "settle_tick_ns": 2,
             "logical_event_count": 1, "native_physical_steps": 1,
             "charged_totals": {}},
        ],
        "boundary_records": boundary_records,
        "audit_records": audit_records,
    }
    if pending:
        result["canonicalization_pending"] = True
    return result


@unittest.skipIf(
    not os.path.isfile(_C4_FRAMES_PATH),
    f"C4 real microframes not found at {_C4_FRAMES_PATH!r}")
class CanonicalizeRawFramesStageBTest(unittest.TestCase):
    """§19.1A Stage B：冻结 C4 009 普通节点流的 V3 archive 重放 → canonical 六域。

    合同对齐（主 agent 裁决 2026-10-06）：audit_records 为数组、
    kind=="raw_v2_frame" 外层记录（内部是 v3 envelope）与
    boundary_records 按提交顺序 1:1 配对（同位序 ordinal 校验）；测试
    fixture 从冻结证据保留普通 node transitions，移除 descriptor 专属
    内容并重排 node-only 序号。fixture_state 精确键集（无额外键），
    资产只经 assets 列表（sha256 验字节）；directory/capacity/history ←
    initial_state 同名字段；causal_metadata ← causal_input；planner_state
    由 hydrate runtime + 冷调度器壳推导冷首边界。
    """

    def setUp(self):
        frozen_frames = json.loads(
            Path(_C4_FRAMES_PATH).read_text(encoding="utf-8"))
        self.assertEqual(frozen_frames[0]["snapshot_kind"], "full")
        self.frames = _stage_b_node_only_v3_frames(frozen_frames[:3])
        self.assertTrue(all(
            frame["schema_version"] == frame["protocol_version"] == 3
            for frame in self.frames))
        self.assertTrue(all(not frame["pipeline_transition_runs"]
                            and not frame["cursor_patches"]
                            for frame in self.frames))
        self.scheduler = _make_scheduler()
        self._tmp = tempfile.TemporaryDirectory(
            prefix="joint-stage-b-test-")
        self.addCleanup(self._tmp.cleanup)
        self.state_dir = Path(self._tmp.name)
        self.fixture_state = _driver_validated_state(
            self.scheduler, self.state_dir)
        # 独立调用经 "root" 键给资产基目录（驱动路径挂对象属性
        # _verified_asset_root——两侧都支持）。
        self.fixture_state["root"] = str(self.state_dir)

    def test_real_frames_replay_to_canonical_views(self):
        import joint.acceptance_projection as ap
        raw = _raw_result(self.frames)
        result = ap.canonicalize_raw_frames(
            raw, self.fixture_state, source_root=_PARENT)
        # 终检 result 为精确键集：canonicalization_pending 已移除且无
        # 额外顶层键。
        self.assertNotIn("canonicalization_pending", result)
        records = result["boundary_records"]
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["stage"], "P")
        self.assertEqual(records[1]["stage"], "D")
        for record, frame in zip(records, self.frames[:2]):
            prestate = record["causal_prestate"]
            self.assertEqual(
                sorted(prestate),
                ["capacity", "directory", "history", "obligations",
                 "resources", "work_ledger"])
            ledger_view = prestate["work_ledger"]
            self.assertFalse(ledger_view["certified_empty"])
            self.assertEqual(
                ledger_view["work_ledger_watermark"],
                frame["logical_seq_end"])
            self.assertEqual(
                prestate["resources"]["backend_snapshot"]["tick_ns"],
                frame["tick"])
            self.assertEqual(
                prestate["resources"]["producer_build"]["binary_sha256"],
                "a" * 64)
            self.assertEqual(
                record["causal_metadata"],
                {"context_tokens": 512, "horizon_tokens": 8})
            self.assertEqual(record["planner_state"]["version"],
                             ap.PLANNER_STATE_VERSION)
        # P planner_state：冷首边界推导（request/session/loads/registries）。
        p_stage = records[0]["planner_state"]["stage_input"]
        self.assertEqual(p_stage["kind"], "P")
        self.assertEqual(
            p_stage["request_view"]["request_id"], "r_stage_b")
        self.assertEqual(
            p_stage["request_view"]["estimated_decode_tokens"], 8)
        self.assertTrue(p_stage["request_view"]["input_kv_bytes_by_tp_rank"])
        # D planner_state：先行 P 边界身份 + 已提交前缀推导。
        d_stage = records[1]["planner_state"]["stage_input"]
        self.assertEqual(d_stage["kind"], "D")
        self.assertEqual(d_stage["boundary_view"]["p_instance_index"], 0)
        self.assertEqual(d_stage["boundary_view"]["p_action"], "stay")
        self.assertEqual(
            d_stage["boundary_view"]["decode_steps"], 8)
        self.assertEqual(d_stage["decision_epoch"], 1)

    def test_positional_ordinal_mismatch_fails_closed(self):
        import joint.acceptance_projection as ap
        raw = _raw_result(self.frames, frame_ordinal_override=7)
        with self.assertRaisesRegex(
                ap.AcceptanceProjectionIncomplete, "commit position"):
            ap.canonicalize_raw_frames(
                raw, self.fixture_state, source_root=_PARENT)

    def test_legacy_dict_audit_records_rejected(self):
        import joint.acceptance_projection as ap
        raw = _raw_result(self.frames)
        raw["audit_records"] = {"raw_v2_frames": raw["audit_records"]}
        with self.assertRaisesRegex(
                ap.AcceptanceProjectionIncomplete, "array"):
            ap.canonicalize_raw_frames(
                raw, self.fixture_state, source_root=_PARENT)

    def test_missing_cold_initial_state_fails_closed(self):
        import joint.acceptance_projection as ap
        raw = _raw_result(self.frames)
        state = {
            key: value for key, value in self.fixture_state.items()
            if key != "root"}
        del state["initial_state"]
        with self.assertRaises(ap.AcceptanceProjectionIncomplete):
            ap.canonicalize_raw_frames(raw, state, source_root=_PARENT)

    def test_refuses_non_pending_input(self):
        import joint.acceptance_projection as ap
        raw = _raw_result(self.frames, pending=False)
        with self.assertRaises(ap.AcceptanceProjectionIncomplete):
            ap.canonicalize_raw_frames(
                raw, self.fixture_state, source_root=_PARENT)

    def test_multi_boundary_d_progression(self):
        """多边界递进（主 agent 指令 2026-10-06）：P + D + D 三边界
        tape——首 D 推导布局，后续 D 以已提交事实递进（epoch 递增、
        义务面刷新、布局一致性校验），全链过驱动终检。"""
        import joint.acceptance_projection as ap
        raw = _raw_result(self.frames)
        third = dict(raw["boundary_records"][1])
        third["boundary_ordinal"] = 3
        raw["boundary_records"].append(third)
        raw["audit_records"].append({
            "kind": "raw_v2_frame", "boundary_ordinal": 3,
            "frame": self.frames[2],
            "runtime_snapshot": {"tick_ns": self.frames[2]["tick"]},
            "producer_build": {
                "binary_sha256": "a" * 64, "kernel_contract": {}}})
        raw["measurements"].append({
            "stage": "D", "fixed_instance_id": "ins1",
            "fixed_action": "stay", "boundary_ordinal": 3,
            "start_tick_ns": 0, "service_tick_ns": 1, "settle_tick_ns": 2,
            "logical_event_count": 1, "native_physical_steps": 1,
            "charged_totals": {}})
        result = ap.canonicalize_raw_frames(
            raw, self.fixture_state, source_root=_PARENT)
        driver = None
        records = result["boundary_records"]
        self.assertEqual(len(records), 3)
        epochs = [r["planner_state"]["stage_input"]["decision_epoch"]
                  for r in records if r["stage"] == "D"]
        self.assertEqual(epochs, [1, 2])
        first_d = records[1]["planner_state"]["stage_input"]["boundary_view"]
        third_d = records[2]["planner_state"]["stage_input"]["boundary_view"]
        for key in ("p_instance_index", "p_action", "history_tokens",
                    "working_bytes_by_tp_rank", "base_bytes_by_tp_rank",
                    "decode_steps", "base_tokens"):
            self.assertEqual(first_d[key], third_d[key], key)
        self.assertNotIn("canonicalization_pending", result)

    def test_increment_layout_without_committed_input_fails_closed(self):
        """增量型 P 前缀：无已提交输入 token 事实（事件带缺席）时
        INCOMPLETE——不虚构 U 拆分。"""
        import joint.acceptance_projection as ap
        raw = _raw_result(self.frames, p_action="remote-read")
        with self.assertRaisesRegex(
                ap.AcceptanceProjectionIncomplete, "input"):
            ap.canonicalize_raw_frames(
                raw, self.fixture_state, source_root=_PARENT)


class RetainedV4OfflineDispatchTest(unittest.TestCase):
    """Retained V4 frames use one strict offline decoder and journal verbatim."""

    def test_hydrate_world_source_selects_v4_and_rejects_mixed_prefix(self):
        from types import SimpleNamespace
        import joint.acceptance_projection as ap

        frame = _empty_v4_retained_frame()
        runtime = SimpleNamespace(
            hardware=SimpleNamespace(mesh_cols=1, mesh_rows=1))
        boundary = {
            "tick_ns": 5,
            "causal_prestate": {"work_ledger": {
                "certified_empty": False,
                # This retained-array key keeps its historical spelling; the
                # child envelope version selects the offline decoder.
                "replay_v3_frames": [frame],
            }},
        }
        source = ap._hydrate_world_source(boundary, runtime)
        self.assertIsNotNone(source)
        self.assertEqual(source._ledger.event_watermark, 0)

        archive_v3 = dict(
            frame, schema_version=3, protocol_version=3,
            bridge_protocol="bridge-v3-20261009",
            work_ledger_schema="workledger-v3-20261009",
            epoch="retained-v3-test",
            replay_token="cpp_online/retained-v3-test/0")
        mixed = dict(boundary)
        mixed["causal_prestate"] = {"work_ledger": {
            "certified_empty": False,
            "replay_v3_frames": [frame, archive_v3],
        }}
        with self.assertRaisesRegex(
                ap.AcceptanceProjectionIncomplete, "mixes V3 archive and V4"):
            ap._hydrate_world_source(mixed, runtime)

    def test_v4_offline_journal_roundtrip_preserves_child_bytes_and_handle(self):
        import joint.acceptance_projection as ap
        from online.bridge_request_journal import iter_request_records
        from online.decision_bridge import BridgeServer

        frame = _empty_v4_retained_frame()
        original = ap._build_dual_ledger_harness(4)
        canonical = original._apply_work_ledger({"work_ledger": frame})
        with tempfile.TemporaryDirectory(prefix="joint-v4-journal-") as temp:
            accepted = {"bytes": canonical}
            server = BridgeServer(
                temp, canonical_request_producer=True,
                work_ledger_bytes_source=lambda _seq: (
                    accepted["bytes"],
                    {"snapshot_kind": frame.get("snapshot_kind"),
                     "event_watermark": None}))
            request = {
                "schema_version": 3,
                "delivery_sequence": 0,
                "tick": 5,
                "reasons": [],
                "arrivals": [],
                "completed_nodes": [],
                "completed_groups": [],
                "affected_ranks": [],
                "event_seqs": [],
                "snapshot_handle": {
                    "epoch": 0, "tick": 5,
                    "kind": "work_ledger_v4"},
                "work_ledger": frame,
            }
            # v11 §8: 帧载体已替换为 shm——测试以 ensure_bridge_shm 充当
            # C++ 生产端铸段;帧负载字节 = 旧 request_0.json 的文件字节。
            from online.bridge_shm import ensure_bridge_shm
            ensure_bridge_shm(temp)
            server.open_transport()
            try:
                server._handle_request(
                    0, lambda _request, _segment: {
                        "kernel_contract": {}},
                    json.dumps(request, separators=(",", ":")).encode("utf-8"))
            finally:
                server._close_audit_streams()
                from online.bridge_shm import clean_stale_bridge_shm
                clean_stale_bridge_shm(temp)

            records = list(iter_request_records(
                temp, attach_work_ledger=True))
            self.assertEqual(len(records), 1)
            _sequence, restored = records[0]
            self.assertEqual(restored["snapshot_handle"]["kind"],
                             "work_ledger_v4")
            self.assertEqual(restored["work_ledger"], frame)
            replay = ap._build_dual_ledger_harness(4)
            replay_canonical = replay._apply_work_ledger({
                "work_ledger": restored["work_ledger"]})
            self.assertEqual(replay_canonical, canonical)
            self.assertEqual(
                replay.joint_pipeline_ledger.snapshot_state(),
                original.joint_pipeline_ledger.snapshot_state())


if __name__ == "__main__":
    unittest.main()
