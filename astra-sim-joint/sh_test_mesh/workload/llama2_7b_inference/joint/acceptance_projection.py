#!/usr/bin/env python3
"""acceptance_projection.py -- 纯验收 data-only 边界重建与 L3 评估入口。

本模块是验收侧专用（执行文档 §19.1 / §19.1A），不新增任何生产实现
选择开关，不含第二套估价/排名逻辑：

* **只读 exporter**：从真实在线对象（sh30 scheduler 运行时构造的
  ``JointCostModel`` 与调度器注册表/账本）导出
  ``joint-accel-planner-state-v1-20261006`` planner_state、
  ``joint-accel-python-runtime-v1-20261006`` runtime asset 与 boundary
  所需的 data-only JSON。导出过程只读，不修改任何在线状态。
* **acceptance-only hydrator**：按原构造器（FaceModel/FaceHardware/
  RemoteMemoryConfig/InferenceGroup/JointMechanismConfig/
  KVCacheManager/JointHardwareRates/LinkFlowRegistry/ServiceFactors/
  InstanceLoadView/SessionKVView/RequestView/PDBoundaryView 与
  sh30 ``_joint_cost_model`` 的同一注入闭包配方）从 JSON 重建原生产
  planner 对象；计价仍走原 ``JointCostModel``/action_work_graph/
  native projection。
* ``evaluate_boundary``：L3 纯验收 API——重建后调用原
  ``select_instance_and_action`` / ``select_decode_target_and_action``，
  返回 ``projection-evaluation-v1-20261006`` DTO。
* ``canonicalize_raw_frames``：冷夹具 raw→canonical 两段接线的
  Stage B（§19.1A）——把同一协议版本的 C++ raw frames（V3 archive 或
  当前 V4 work-ledger envelope）经严格离线双账本 reader 重放，导出规范
  因果前态六域与 planner_state。

失败语义：缺真实能力/输入 → ``AcceptanceProjectionIncomplete``
（调用方转 INCOMPLETE）；真实 backend/预算失败 → 保留原始异常并按
FAIL 落 failure DTO（只 version/status/reason + 已知身份，不填空
成功数组）。
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

DTO_VERSION = "projection-evaluation-v1-20261006"
PLANNER_STATE_VERSION = "joint-accel-planner-state-v1-20261006"
RUNTIME_ASSET_VERSION = "joint-accel-python-runtime-v1-20261006"
NO_EXOGENOUS_GENERATOR_IDENTITY = (
    "no-exogenous-generator/v6-compact-window-20261005")
BOUNDARY_SIX_DOMAINS = (
    "work_ledger", "resources", "directory", "capacity",
    "obligations", "history")
_STAGE_VALUES = ("P", "D")


class AcceptanceProjectionIncomplete(RuntimeError):
    """缺真实能力/输入（INCOMPLETE 类；不伪造测量）。"""


class AcceptanceProjectionFailure(RuntimeError):
    """真实 backend/预算/一致性失败（FAIL 类；保留原始证据）。"""


# ------------------------------------------------------------- 通用工具 --


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode("utf-8")


def _semantic_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _sha256_file(path: os.PathLike | str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_keys(value: Any, keys: Sequence[str], *, where: str) -> None:
    if not isinstance(value, dict):
        raise AcceptanceProjectionIncomplete(
            f"{where} must be a JSON object, got {type(value).__name__}")
    missing = [name for name in keys if name not in value]
    if missing:
        raise AcceptanceProjectionIncomplete(
            f"{where} is missing required keys: {missing}")


def _jsonify(value: Any) -> Any:
    """确定性 JSON 投影（与 sh30 probe 视图同族：事实值全保留）。"""
    import collections.abc
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, collections.abc.Mapping):
        return {str(key): _jsonify(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_jsonify(item) for item in value]
    if hasattr(value, "__dataclass_fields__"):
        import dataclasses
        return _jsonify(dataclasses.asdict(value))
    if hasattr(value, "as_dict") and callable(value.as_dict):
        return _jsonify(value.as_dict())
    return str(value)


# ============================================================== exporter ==


def export_face_model(model) -> dict:
    """FaceModel 语义字段（model_exact_keys；无 provenance）。"""
    return {
        "layers": int(model.layers),
        "hidden_size": int(model.hidden_size),
        "ffn_size": int(model.ffn_size),
        "num_heads": int(model.num_heads),
        "vocab_size": int(model.vocab_size),
        "bytes_per_elem": int(model.bytes_per_elem),
        "mlp_variant": str(model.mlp_variant),
    }


def export_face_hardware(hardware) -> dict:
    """FaceHardware 语义字段（hardware_exact_keys）。"""
    return {
        "mesh_rows": int(hardware.mesh_rows),
        "mesh_cols": int(hardware.mesh_cols),
        "local_hbm_capacity_bytes": int(hardware.local_hbm_capacity_bytes),
        "local_hbm_bandwidth_gbps": float(
            hardware.local_hbm_bandwidth_gbps),
        "d2d_bandwidth_gbps": float(hardware.d2d_bandwidth_gbps),
        "peak_perf_tflops": float(hardware.peak_perf_tflops),
        "d2d_latency_ns": int(hardware.d2d_latency_ns),
        "local_hbm_latency_ns": int(hardware.local_hbm_latency_ns),
        "label": str(hardware.label),
    }


def export_remote_memory_view(scheduler) -> dict:
    """remote_memory 语义字段（remote_memory_exact_keys，无 path）。

    真实 ``__init__`` 装配的调度器持有 remote_memory 配置对象时从其
    导出；``__new__`` 测试壳/直构夹具从在线 ``_joint_rates`` 反解池速
    率（决策时点真值，data-only）。edge_npus 取 kv_manager.edge_ranks
    （nearest_edge 的真实输入）。
    """
    remote_memory = getattr(scheduler, "_joint_remote_memory_config", None)
    rates = getattr(scheduler, "_joint_rates", None)
    if remote_memory is not None:
        edge_npus = tuple(int(item) for item in remote_memory.edge_npus)
        pool_gbps = float(remote_memory.remote_mem_bw_gbps)
        pool_latency = int(remote_memory.remote_mem_latency_ns)
        mesh_rows = int(remote_memory.mesh_rows)
        mesh_cols = int(remote_memory.mesh_cols)
        logical_pool = str(remote_memory.logical_pool)
    else:
        if rates is None:
            raise AcceptanceProjectionIncomplete(
                "scheduler exposes neither a remote-memory config nor "
                "_joint_rates; cannot export remote_memory semantics")
        manager = getattr(scheduler, "kv_manager", None)
        edge_ranks = getattr(manager, "edge_ranks", None)
        if edge_ranks is None:
            from face_scheduler import physical_edge_ranks
            edge_ranks = physical_edge_ranks(scheduler.topology.hardware)
        edge_npus = tuple(int(item) for item in edge_ranks)
        pool_gbps = float(rates.pool_port_bytes_per_ns)
        pool_latency = int(rates.pool_latency_ns)
        hardware = scheduler.topology.hardware
        mesh_rows = int(hardware.mesh_rows)
        mesh_cols = int(hardware.mesh_cols)
        logical_pool = ""
    return {
        "edge_npus": list(edge_npus),
        "mesh_rows": mesh_rows,
        "mesh_cols": mesh_cols,
        "remote_mem_latency_ns": pool_latency,
        "remote_mem_bw_gbps": pool_gbps,
        "logical_pool": logical_pool,
    }


def export_runtime_asset(scheduler) -> dict:
    """``joint-accel-python-runtime-v1-20261006`` runtime asset（只读）。

    从真实在线对象导出原构造器数据字段；source/env/output/request
    queue 路径与 trace 平均 decode 长度不进入语义对象。
    """
    hardware = scheduler.topology.hardware
    model = scheduler.model
    joint_config = scheduler.joint_config
    die_context = getattr(scheduler, "die_model_context", None)
    # 驱动 validate_fixture_state 的封闭取值集：norm | die（"roofline" 是
    # 仓内习惯名，wire 合同用 norm）。
    compute_cost_model = "die" if die_context is not None else "norm"
    return {
        "version": RUNTIME_ASSET_VERSION,
        "face_trace": {
            "model": export_face_model(model),
            "model_name": str(getattr(scheduler, "model_name", "")
                              or ""),
            "position_encoding": str(
                getattr(scheduler, "position_encoding", "") or ""),
            "remote_operand_loads": bool(
                getattr(scheduler, "remote_operand_loads", False)),
            "kv_reserve_context_tokens": int(
                getattr(scheduler, "kv_reserve_context_tokens", 0)),
            "prefill_chunk_size": int(scheduler.p_chunk),
            "trace_granularity": str(
                getattr(scheduler, "trace_granularity", "") or ""),
            "hardware_capacity_profile": str(
                getattr(scheduler, "hardware_capacity_profile", "") or ""),
        },
        "hardware": export_face_hardware(hardware),
        "remote_memory": export_remote_memory_view(scheduler),
        "inference_groups": [
            {
                "name": str(instance.name),
                "pg_name": str(instance.pg_name),
                "ranks": [int(rank) for rank in instance.ranks],
            }
            for instance in scheduler.topology.instances
        ],
        "joint_config": {
            "category_mode": str(joint_config.category_mode),
            "scheduler_mode": str(joint_config.scheduler_mode),
            "layer_policy": str(joint_config.layer_policy),
            "remote_actions": str(joint_config.remote_actions),
            "remote_read_tile_bytes": str(joint_config.remote_read_tile_bytes),
            "remote_read_partial": str(joint_config.remote_read_partial),
            "allowed_actions": str(joint_config.allowed_actions),
            "quota_mode": str(joint_config.quota_mode),
            "boundary_policy": str(joint_config.boundary_policy),
            "combo": joint_config.combo,
        },
        "system_backend": {
            "compute_cost_model": compute_cost_model,
            "die_profile": (
                str(getattr(scheduler, "die_profile_name", "") or "")
                if die_context is not None else None),
            "die_model_config_asset_id": None,
        },
    }


def export_capacity_summary_view(summary) -> dict:
    """InstanceCapacitySummary 逐桶事实投影（capacity_summary 合同形态）。"""
    if summary is None:
        return None
    buckets = []
    for bucket in getattr(summary, "buckets", ()):
        key = bucket.key
        buckets.append({
            "category": key.category,
            "suffix_class": key.suffix_class,
            "shard_ratio": list(key.shard_ratio),
            "writeback_ports": list(key.writeback_ports),
            "reclaimable_by_rank": list(bucket.reclaimable_by_rank),
            "clean_by_rank": list(bucket.clean_by_rank),
            "dirty_by_rank": list(bucket.dirty_by_rank),
            "atom_count_by_rank": list(bucket.atom_count_by_rank),
            "max_atom_by_rank": list(bucket.max_atom_by_rank),
        })
    return {
        "instance_index": int(summary.instance_index),
        "snapshot_version": int(summary.snapshot_version),
        "maintenance_event_count": int(
            summary.maintenance_event_count),
        "error_sources": list(summary.error_sources),
        "buckets": buckets,
    }


def export_instance_load_view(load) -> dict:
    """InstanceLoadView 的 instance_loads_item_exact_keys 投影。"""
    return {
        "instance_index": int(load.instance_index),
        "queued_task_load_ns": int(load.queued_task_load_ns),
        "running_task_load_ns": int(load.running_task_load_ns),
        "active_decode_task_load_ns": int(
            load.active_decode_task_load_ns),
        "hbm_remaining_bytes_by_tp_rank": [
            int(value) for value in load.hbm_remaining_bytes_by_tp_rank],
        "reclaimable_bytes_by_tp_rank": [
            int(value) for value in load.reclaimable_bytes_by_tp_rank],
        "capacity_summary": export_capacity_summary_view(
            load.capacity_summary),
        "capacity_ledger_by_tp_rank": [
            [int(value) for value in row]
            for row in load.capacity_ledger_by_tp_rank],
    }


def export_service_factors(factors) -> dict:
    """ServiceFactors 的 service_factors_exact_keys 投影。"""
    return {
        "prefill_factor": float(factors.prefill_factor),
        "decode_factor": float(factors.decode_factor),
        "transfer_factor": float(factors.transfer_factor),
        "updates": {
            str(key): int(value)
            for key, value in getattr(factors, "updates", {}).items()},
        "rejected": {
            str(key): int(value)
            for key, value in getattr(factors, "rejected", {}).items()},
        "anomalous": {
            str(key): int(value)
            for key, value in getattr(factors, "anomalous", {}).items()},
        "contended": {
            str(key): int(value)
            for key, value in getattr(factors, "contended", {}).items()},
    }


def export_registries(scheduler, cost_model=None) -> dict:
    """registries_exact_keys 投影（决策时点真值）。"""
    flows = getattr(scheduler, "_joint_flows", None)
    if flows is None and cost_model is not None:
        flows = cost_model.flow_registry
    noc_flows = []
    if flows is not None:
        snapshot = flows.snapshot()
        for key, count in sorted(snapshot.items()):
            source, destination = key.split("->")
            noc_flows.append({
                "source_rank": int(source),
                "destination_rank": int(destination),
                "count": int(count),
            })
    pool_ports = getattr(scheduler, "_pool_ports", None)
    pool_edge_counts = []
    if pool_ports is not None:
        for edge_rank, count in sorted(
                getattr(pool_ports, "_counts", {}).items()):
            pool_edge_counts.append({
                "edge_rank": int(edge_rank), "count": int(count)})
    hbm_ports = getattr(scheduler, "_hbm_ports", None)
    hbm_port_counts = []
    if hbm_ports is not None:
        active_provider = getattr(hbm_ports, "_active_decode_provider", None)
        known = set(getattr(hbm_ports, "_counts", {}))
        for rank in sorted(known):
            registered = int(hbm_ports._counts.get(rank, 0))
            active = 0
            if active_provider is not None:
                active = max(0, int(active_provider(int(rank))))
            hbm_port_counts.append({
                "rank": int(rank),
                "registered_transfer_flows": registered,
                "active_decode_streams": active,
            })
    rates = getattr(scheduler, "_link_telemetry_rates", None) or {}
    flow_counts = getattr(
        scheduler, "_link_telemetry_flow_counts", None) or {}
    window = getattr(scheduler, "_telemetry_last_window_bounds_ns", None)
    telemetry = {
        "rates": {
            str(key): float(value) for key, value in rates.items()},
        "flow_counts": {
            str(key): float(value)
            for key, value in flow_counts.items()},
        # 驱动 planner_state 合同为单一 int（窗口时长 ns）；无窗口 = 0
        #（显式事实，非 None——终检 _require_int）。
        "window_ns": (
            int(window[1]) - int(window[0])
            if window is not None else 0),
    }
    forecast_history = _export_forecast_history(scheduler)
    return {
        "service_factors": export_service_factors(
            cost_model.service_factors if cost_model is not None
            else getattr(scheduler, "_joint_factors")),
        "noc_flows": noc_flows,
        "pool_edge_counts": pool_edge_counts,
        "hbm_port_counts": hbm_port_counts,
        "telemetry": telemetry,
        "forecast_history": forecast_history,
    }


def _export_forecast_history(scheduler) -> list:
    """registries.forecast_history：外生预测引擎的因果事实面（数组形态
    ——驱动 planner_state 合同要求 array；空数组 = 无外生引擎的显式
    事实，非缺省冒充）。"""
    engine = None
    try:
        engine = scheduler._joint_forecast_engine_or_default()
    except Exception:
        engine = getattr(scheduler, "_joint_forecast_engine", None)
    if engine is None:
        return []
    return [{"engine": _engine_stable_projection(engine)}]


def _engine_stable_projection(engine) -> Any:
    """engine 的稳定 JSON 投影。

    可结构化导出（dataclass/as_dict 面）时保留事实值；否则输出固定
    token ``<engine-object>``。``str(engine)`` 含进程地址，会使双副本
    确定性比较永远失败，禁止进入 canonical 结果。"""
    if hasattr(engine, "__dataclass_fields__") or (
            hasattr(engine, "as_dict") and callable(engine.as_dict)):
        return _jsonify(engine)
    return "<engine-object>"


def export_forecast_generator(
        *,
        context_tokens: int,
        horizon_tokens: int,
        generator_identity: Optional[str] = None,
        generator_inputs: Optional[Mapping[str, Any]] = None,
) -> dict:
    """forecast_generator 严格三键（§19.1）。

    无外生生成器时 identity 固定 ``no-exogenous-generator/
    v6-compact-window-20261005``、generator_inputs 为空 mapping。
    """
    identity = (
        generator_identity
        if generator_identity else NO_EXOGENOUS_GENERATOR_IDENTITY)
    inputs = dict(generator_inputs) if generator_inputs else {}
    causal_inputs = {
        "context_tokens": int(context_tokens),
        "horizon_tokens": int(horizon_tokens),
        "generator_inputs": inputs,
    }
    if causal_inputs["context_tokens"] < 0:
        raise AcceptanceProjectionIncomplete(
            "forecast context_tokens must be a non-negative integer")
    if causal_inputs["horizon_tokens"] <= 0:
        raise AcceptanceProjectionIncomplete(
            "forecast horizon_tokens must be a positive integer")
    return {
        "identity": identity,
        "causal_inputs": causal_inputs,
        "semantic_sha256": _semantic_sha256(
            {"identity": identity, "causal_inputs": causal_inputs}),
    }


def export_planner_state_P(
        scheduler, cost_model, *, session, request,
        prep_frontier_by_instance: Optional[Mapping[int, int]],
        remote_enabled: bool,
        context_tokens: int,
        horizon_tokens: int,
        generator_identity: Optional[str] = None,
        generator_inputs: Optional[Mapping[str, Any]] = None,
) -> dict:
    """P 边界 planner_state（joint-accel-planner-state-v1-20261006）。"""
    prep = {
        str(int(index)): int(value)
        for index, value in (prep_frontier_by_instance or {}).items()}
    return {
        "version": PLANNER_STATE_VERSION,
        "stage_input": {
            "kind": "P",
            "session_view": {
                "session_id": session.session_id,
                "home_instance": session.home_instance,
                "resident_instance": session.resident_instance,
                "location": session.location,
                "history_tokens": int(session.history_tokens),
                "resident_prefix_layers": int(
                    session.resident_prefix_layers),
                "history_bytes_by_tp_rank": [
                    int(value) for value in session.history_bytes_by_tp_rank],
                "missing_bytes_by_tp_rank": [
                    int(value) for value in session.missing_bytes_by_tp_rank],
            },
            "request_view": {
                "request_id": request.request_id,
                "session_id": request.session_id,
                "input_tokens": int(request.input_tokens),
                "history_tokens_before": int(
                    request.history_tokens_before),
                "estimated_decode_tokens": int(
                    request.estimated_decode_tokens),
                "horizon_source": request.horizon_source,
                "input_kv_bytes_by_tp_rank": [
                    int(value)
                    for value in request.input_kv_bytes_by_tp_rank],
                "prefill_scan_passes": request.prefill_scan_passes,
                "prefill_chunk_tokens": request.prefill_chunk_tokens,
                "query_tile_tokens": request.query_tile_tokens,
            },
            "prep_frontier_by_instance": prep,
            "remote_enabled": bool(remote_enabled),
        },
        "instance_loads": [
            export_instance_load_view(load)
            for _, load in sorted(cost_model.loads.items())],
        "registries": export_registries(scheduler, cost_model),
        "forecast_generator": export_forecast_generator(
            context_tokens=context_tokens,
            horizon_tokens=horizon_tokens,
            generator_identity=generator_identity,
            generator_inputs=generator_inputs),
    }


def export_planner_state_D(
        scheduler, cost_model, *, boundary_view,
        runtime_state: Mapping[str, Any],
        decision_epoch: int,
        inherited_obligation_ids: Sequence[str],
        remote_enabled: bool,
        context_tokens: int,
        horizon_tokens: int,
        generator_identity: Optional[str] = None,
        generator_inputs: Optional[Mapping[str, Any]] = None,
) -> dict:
    """D 边界 planner_state（严格字段同合同）。"""
    obligations = []
    for item in runtime_state.get("obligations", ()):
        obligations.append({
            "obligation_id": item["obligation_id"],
            "kind": item["kind"],
            "receiver_instance": item["receiver_instance"],
            "replica_id": item["replica_id"],
            "summary": _jsonify(item.get("summary")),
        })
    first_token = runtime_state.get("first_token") or {}
    return {
        "version": PLANNER_STATE_VERSION,
        "stage_input": {
            "kind": "D",
            "boundary_view": {
                "session_id": boundary_view.session_id,
                "home_instance": boundary_view.home_instance,
                "p_instance_index": int(boundary_view.p_instance_index),
                "p_action": boundary_view.p_action,
                "history_tokens": int(boundary_view.history_tokens),
                "working_bytes_by_tp_rank": [
                    int(value)
                    for value in boundary_view.working_bytes_by_tp_rank],
                "base_bytes_by_tp_rank": [
                    int(value)
                    for value in boundary_view.base_bytes_by_tp_rank],
                "decode_steps": int(boundary_view.decode_steps),
                "base_tokens": int(boundary_view.base_tokens),
                "kv_token_bytes": boundary_view.kv_token_bytes,
                "logic_home_instance": boundary_view.logic_home_instance,
                "kv_token_bytes_by_tp_rank": (
                    None if boundary_view.kv_token_bytes_by_tp_rank is None
                    else [int(value) for value in
                          boundary_view.kv_token_bytes_by_tp_rank]),
                "working_intervals": [
                    list(interval) for interval in
                    boundary_view.working_intervals],
            },
            "runtime_state": {
                "terminated": bool(runtime_state.get("terminated", False)),
                "first_token": {
                    "issued": bool(first_token.get("issued", False)),
                    "consumed": bool(first_token.get("consumed", False)),
                    "rng_advances": int(first_token.get("rng_advances", 0)),
                    "rebuild_sampling_suppressed": bool(
                        first_token.get("rebuild_sampling_suppressed",
                                        False)),
                },
                "obligations": obligations,
                "working_layers": _jsonify(
                    runtime_state.get("working_layers")),
                "base_layers": _jsonify(runtime_state.get("base_layers")),
            },
            "decision_epoch": int(decision_epoch),
            "inherited_obligation_ids": [
                str(item) for item in inherited_obligation_ids],
            "remote_enabled": bool(remote_enabled),
        },
        "instance_loads": [
            export_instance_load_view(load)
            for _, load in sorted(cost_model.loads.items())],
        "registries": export_registries(scheduler, cost_model),
        "forecast_generator": export_forecast_generator(
            context_tokens=context_tokens,
            horizon_tokens=horizon_tokens,
            generator_identity=generator_identity,
            generator_inputs=generator_inputs),
    }


def export_causal_prestate(scheduler, *, focus_session_id: str) -> dict:
    """六域 causal_prestate（规范 logical 视图，只读）。

    * work_ledger：普通账本 ``export_state()`` + pipeline 账本
      ``remaining_view()`` 的合成规范逻辑剩余视图；双账本均未装载时
      显式 certified_empty（事实，不是缺省冒充）。
    * resources：Python 运行时真实拥有的物理供给状态（速率 + 注册表
      在途计数）；C++ 后端采样帧在场时由调用方覆盖（Stage B）。
    * directory/capacity：复用 sh30 probe 只读事实投影。
    * obligations：P 边界恒空（生产口径）；D 边界由调用方注入。
    * history：horizon 估计器 + 条件长度估计器的充分统计。
    """
    ledger = getattr(scheduler, "joint_work_ledger", None)
    pipeline = getattr(scheduler, "joint_pipeline_ledger", None)
    ordinary = None
    pipeline_view = None
    if ledger is not None and (
            ledger.full_load_count > 0 or ledger.delta_load_count > 0):
        ordinary = ledger.export_state()
    if pipeline is not None and getattr(pipeline, "initialized", False):
        pipeline_view = _jsonify(pipeline.remaining_view())
    if ordinary is None and pipeline_view is None:
        work_ledger_view = {
            "certified_empty": True,
            "reason": "no protocol ledger loaded at this boundary "
                      "(production absent-path disclosure)",
        }
    else:
        work_ledger_view = {
            "certified_empty": False,
            "ordinary": ordinary,
            "pipeline": pipeline_view,
        }
    rates = getattr(scheduler, "_joint_rates", None)
    resources = {
        "rates": rates.as_dict() if rates is not None else None,
        "noc_flows": export_registries(scheduler)["noc_flows"],
        "pool_edge_counts": export_registries(scheduler)[
            "pool_edge_counts"],
        "hbm_port_counts": export_registries(scheduler)[
            "hbm_port_counts"],
        "backend_snapshot": None,
    }
    directory = scheduler._joint_probe_directory_view()
    capacity = scheduler._joint_probe_capacity_view()
    history = _export_history_view(scheduler, focus_session_id)
    return {
        "work_ledger": work_ledger_view,
        "resources": resources,
        "directory": directory,
        "capacity": capacity,
        "obligations": [],
        "history": history,
    }


def _export_history_view(scheduler, focus_session_id: str) -> dict:
    """history 域：horizon + 条件长度估计的 run/focus 充分统计。"""
    horizon = getattr(scheduler, "_joint_horizon", None)
    horizon_view = None
    if horizon is not None:
        horizon_view = {
            "cold_start_default_forward_steps": int(
                horizon._cold_default),
            "session_stats": {
                str(key): [int(value[0]), int(value[1])]
                for key, value in horizon._session_stats.items()},
            "run_count": int(horizon._run_count),
            "run_sum": int(horizon._run_sum),
            "version": int(horizon.version),
        }
    length_history = None
    try:
        estimator = scheduler._joint_length_history_or_default()
        length_history = {
            "run": _level_stats_view(estimator._run_stats),
            "focus_session": _level_stats_view(
                estimator._session_stats.get(focus_session_id)),
            "focus_category": None,
            "run_disclosure": estimator.as_dict(),
        }
    except Exception:
        length_history = None
    return {
        "horizon_estimator": horizon_view,
        "conditional_length": length_history,
    }


def _level_stats_view(stats) -> Optional[dict]:
    """_LevelStats 全充分统计（count/total_sum/max_observed/bucket_counts/
    tail_count/tail_sum/censored_count/censored_progress_sum）。"""
    if stats is None:
        return None
    return {
        "count": int(stats.count),
        "total_sum": int(stats.total_sum),
        "max_observed": int(stats.max_observed),
        "bucket_counts": [int(value) for value in stats.bucket_counts],
        "tail_count": int(stats.tail_count),
        "tail_sum": int(stats.tail_sum),
        "censored_count": int(stats.censored_count),
        "censored_progress_sum": int(stats.censored_progress_sum),
    }


def export_boundary(
        scheduler, cost_model, *,
        request_id: str,
        stage: str,
        boundary_ordinal: int,
        tick_ns: int,
        fixed_instance_id,
        fixed_action,
        focus_session_id: str,
        planner_state: Mapping[str, Any],
        causal_metadata: Mapping[str, Any],
        obligations: Optional[Sequence[Mapping[str, Any]]] = None,
) -> dict:
    """boundary_record 字面九字段（§19.1）。"""
    if stage not in _STAGE_VALUES:
        raise AcceptanceProjectionIncomplete(
            f"boundary stage must be one of {_STAGE_VALUES}, got {stage!r}")
    prestate = export_causal_prestate(
        scheduler, focus_session_id=focus_session_id)
    if obligations:
        prestate["obligations"] = [
            _jsonify(item) for item in obligations]
    return {
        "request_id": str(request_id),
        "stage": stage,
        "boundary_ordinal": int(boundary_ordinal),
        "tick_ns": int(tick_ns),
        "fixed_instance_id": fixed_instance_id,
        "fixed_action": fixed_action,
        "causal_prestate": prestate,
        "causal_metadata": dict(causal_metadata),
        "planner_state": dict(planner_state),
    }


# ============================================================== hydrator ==


@dataclass
class HydratedRuntime:
    """原构造器重建的运行时基底（无第二套公式）。"""
    hardware: Any
    model: Any
    topology: Any
    joint_config: Any
    remote_memory: Any
    rates: Any
    p_chunk: int
    kv_manager: Any
    edge_ranks: tuple

    @property
    def instance_indices(self) -> tuple:
        return tuple(sorted(instance.index for instance
                            in self.topology.instances))


def hydrate_runtime_asset(asset: Mapping[str, Any],
                          *,
                          private_dir: os.PathLike | str) -> HydratedRuntime:
    """按原构造器从 runtime asset JSON 重建运行时基底。"""
    _require_keys(
        asset,
        ("version", "face_trace", "hardware", "remote_memory",
         "inference_groups", "joint_config", "system_backend"),
        where="python_runtime_asset")
    if asset["version"] != RUNTIME_ASSET_VERSION:
        raise AcceptanceProjectionIncomplete(
            f"runtime asset version {asset['version']!r} != "
            f"{RUNTIME_ASSET_VERSION!r}")
    from face_scheduler import (
        FaceHardware,
        FaceInstanceSpec,
        FaceModel,
        KVCacheManager,
        build_instances,
    )
    from generate_trace import InferenceGroup, RemoteMemoryConfig
    from joint.joint_config import JointMechanismConfig
    from joint.joint_cost_model import JointHardwareRates

    face_trace = asset["face_trace"]
    model_fields = face_trace["model"]
    model = FaceModel(
        layers=int(model_fields["layers"]),
        hidden_size=int(model_fields["hidden_size"]),
        ffn_size=int(model_fields["ffn_size"]),
        num_heads=int(model_fields["num_heads"]),
        vocab_size=int(model_fields["vocab_size"]),
        bytes_per_elem=int(model_fields["bytes_per_elem"]),
        mlp_variant=str(model_fields["mlp_variant"]),
    )
    hardware_fields = asset["hardware"]
    hardware = FaceHardware(
        mesh_rows=int(hardware_fields["mesh_rows"]),
        mesh_cols=int(hardware_fields["mesh_cols"]),
        local_hbm_capacity_bytes=int(
            hardware_fields["local_hbm_capacity_bytes"]),
        local_hbm_bandwidth_gbps=float(
            hardware_fields["local_hbm_bandwidth_gbps"]),
        d2d_bandwidth_gbps=float(hardware_fields["d2d_bandwidth_gbps"]),
        peak_perf_tflops=float(hardware_fields["peak_perf_tflops"]),
        d2d_latency_ns=int(hardware_fields["d2d_latency_ns"]),
        local_hbm_latency_ns=int(
            hardware_fields["local_hbm_latency_ns"]),
        label=str(hardware_fields.get("label", "FACE")),
    )
    groups = []
    specs = []
    for group_fields in asset["inference_groups"]:
        group = InferenceGroup(
            name=str(group_fields["name"]),
            pg_name=str(group_fields["pg_name"]),
            ranks=tuple(int(rank) for rank in group_fields["ranks"]))
        groups.append(group)
        specs.append(FaceInstanceSpec(
            name=group.name, pg_name=group.pg_name, ranks=group.ranks))
    topology = build_instances(hardware, tuple(specs))
    remote_fields = asset["remote_memory"]
    # RemoteMemoryConfig.path 由已验证语义字段在仓外私有目录派生（合同：
    # 不消费资产内绝对路径）。roofline 后端不读该文件。
    private_path = Path(private_dir) / "remote_memory_semantics.json"
    private_path.parent.mkdir(parents=True, exist_ok=True)
    if not private_path.exists():
        private_path.write_text(json.dumps(
            remote_fields, sort_keys=True, indent=1), encoding="utf-8")
    remote_memory = RemoteMemoryConfig(
        path=private_path,
        edge_npus=tuple(int(item) for item in remote_fields["edge_npus"]),
        mesh_rows=int(remote_fields["mesh_rows"]),
        mesh_cols=int(remote_fields["mesh_cols"]),
        remote_mem_latency_ns=int(remote_fields["remote_mem_latency_ns"]),
        remote_mem_bw_gbps=float(remote_fields["remote_mem_bw_gbps"]),
        logical_pool=str(remote_fields["logical_pool"]),
    )
    config_fields = asset["joint_config"]
    joint_config = JointMechanismConfig(
        category_mode=str(config_fields["category_mode"]),
        scheduler_mode=str(config_fields["scheduler_mode"]),
        layer_policy=str(config_fields["layer_policy"]),
        remote_actions=str(config_fields["remote_actions"]),
        remote_read_tile_bytes=str(
            config_fields["remote_read_tile_bytes"]),
        remote_read_partial=str(config_fields["remote_read_partial"]),
        allowed_actions=str(config_fields["allowed_actions"]),
        quota_mode=str(config_fields["quota_mode"]),
        boundary_policy=str(config_fields["boundary_policy"]),
        combo=config_fields.get("combo"),
    )
    rates = JointHardwareRates.from_gbps(
        noc_link_gbps=hardware.d2d_bandwidth_gbps,
        pool_port_gbps=remote_memory.remote_mem_bw_gbps,
        local_hbm_gbps=hardware.local_hbm_bandwidth_gbps,
        d2d_latency_ns=int(hardware.d2d_latency_ns),
        pool_latency_ns=int(remote_memory.remote_mem_latency_ns),
    )
    backend = asset["system_backend"]
    if backend.get("compute_cost_model") == "die":
        raise AcceptanceProjectionIncomplete(
            "die-backend runtime hydration requires the SHA256-verified "
            "portable die_model_config_asset (die_model_config_asset_id="
            f"{backend.get('die_model_config_asset_id')!r}); not exercised "
            "by the current bounded proof — refusing to synthesize a die "
            "profile")
    kv_manager = KVCacheManager(
        topology, model,
        edge_ranks=remote_memory.edge_npus,
        category_mode=joint_config.category_mode,
        layer_policy=joint_config.layer_policy,
        pool_bandwidth_gbps=remote_memory.remote_mem_bw_gbps,
        pool_latency_ns=remote_memory.remote_mem_latency_ns,
    )
    return HydratedRuntime(
        hardware=hardware,
        model=model,
        topology=topology,
        joint_config=joint_config,
        remote_memory=remote_memory,
        rates=rates,
        p_chunk=int(face_trace["prefill_chunk_size"]),
        kv_manager=kv_manager,
        edge_ranks=tuple(int(item) for item in remote_memory.edge_npus),
    )


def _hydrate_capacity_summary(fields) -> Any:
    """InstanceCapacitySummary 原构造器重建（capacity_summary JSON）。"""
    from joint.capacity_summary import (
        CapacityBucket,
        CapacityBucketKey,
        InstanceCapacitySummary,
    )
    if fields is None:
        return None
    buckets = []
    for bucket_fields in fields.get("buckets", ()):
        key = CapacityBucketKey(
            category=str(bucket_fields["category"]),
            suffix_class=str(bucket_fields["suffix_class"]),
            shard_ratio=tuple(
                int(value) for value in bucket_fields["shard_ratio"]),
            writeback_ports=tuple(
                int(value) for value in bucket_fields["writeback_ports"]),
        )
        buckets.append(CapacityBucket(
            key=key,
            reclaimable_by_rank=tuple(
                int(value)
                for value in bucket_fields["reclaimable_by_rank"]),
            clean_by_rank=tuple(
                int(value) for value in bucket_fields["clean_by_rank"]),
            dirty_by_rank=tuple(
                int(value) for value in bucket_fields["dirty_by_rank"]),
            atom_count_by_rank=tuple(
                int(value)
                for value in bucket_fields["atom_count_by_rank"]),
            max_atom_by_rank=tuple(
                int(value)
                for value in bucket_fields["max_atom_by_rank"]),
        ))
    return InstanceCapacitySummary(
        instance_index=int(fields["instance_index"]),
        buckets=tuple(buckets),
        snapshot_version=int(fields.get("snapshot_version", 0)),
        maintenance_event_count=int(
            fields.get("maintenance_event_count", 0)),
        error_sources=tuple(
            str(item) for item in fields.get("error_sources", ())),
    )


def hydrate_instance_loads(planner_state: Mapping[str, Any]) -> dict:
    """InstanceLoadView 集合（原构造器；含容量摘要重建）。"""
    from joint.joint_cost_model import InstanceLoadView
    loads = {}
    for fields in planner_state.get("instance_loads", ()):
        loads[int(fields["instance_index"])] = InstanceLoadView(
            instance_index=int(fields["instance_index"]),
            queued_task_load_ns=int(fields["queued_task_load_ns"]),
            running_task_load_ns=int(fields["running_task_load_ns"]),
            active_decode_task_load_ns=int(
                fields["active_decode_task_load_ns"]),
            hbm_remaining_bytes_by_tp_rank=tuple(
                int(value)
                for value in fields["hbm_remaining_bytes_by_tp_rank"]),
            reclaimable_bytes_by_tp_rank=tuple(
                int(value)
                for value in fields["reclaimable_bytes_by_tp_rank"]),
            capacity_summary=_hydrate_capacity_summary(
                fields.get("capacity_summary")),
            capacity_ledger_by_tp_rank=tuple(
                tuple(int(value) for value in row)
                for row in fields.get("capacity_ledger_by_tp_rank", ())),
        )
    return loads


def hydrate_registries(planner_state: Mapping[str, Any]) -> dict:
    """registries → 原构造器对象（注册表/因子/遥测）。

    返回 {"flow_registry", "service_factors", "pool_counts",
    "hbm_counts", "telemetry_rates", "telemetry_flow_counts",
    "telemetry_window"}。
    """
    from joint.joint_cost_model import LinkFlowRegistry, ServiceFactors
    registries = planner_state.get("registries")
    if not isinstance(registries, dict):
        raise AcceptanceProjectionIncomplete(
            "planner_state.registries is required for hydration")
    flow_registry = LinkFlowRegistry()
    for item in registries.get("noc_flows", ()):
        for _ in range(int(item["count"])):
            flow_registry.register(
                int(item["source_rank"]), int(item["destination_rank"]))
    factors_fields = registries["service_factors"]
    service_factors = ServiceFactors(
        prefill_factor=float(factors_fields["prefill_factor"]),
        decode_factor=float(factors_fields["decode_factor"]),
        transfer_factor=float(factors_fields["transfer_factor"]),
        updates={str(key): int(value) for key, value in
                 factors_fields.get("updates", {}).items()},
        rejected={str(key): int(value) for key, value in
                  factors_fields.get("rejected", {}).items()},
        anomalous={str(key): int(value) for key, value in
                   factors_fields.get("anomalous", {}).items()},
        contended={str(key): int(value) for key, value in
                   factors_fields.get("contended", {}).items()},
    )
    pool_counts = {
        int(item["edge_rank"]): int(item["count"])
        for item in registries.get("pool_edge_counts", ())}
    hbm_counts = {
        int(item["rank"]): (int(item["registered_transfer_flows"]),
                            int(item["active_decode_streams"]))
        for item in registries.get("hbm_port_counts", ())}
    telemetry = registries.get("telemetry") or {}
    rates = telemetry.get("rates") or {}
    flow_counts = telemetry.get("flow_counts") or {}
    # wire 为单一 int 时长；0 = 无窗口（缺省事实）。非零时长无法还原
    # (start, end) 边界——JCM 的窗口披露位保持缺省（None），不虚构。
    window = telemetry.get("window_ns")
    parsed_rates = {}
    for key, value in rates.items():
        parsed_rates[_telemetry_key(key)] = float(value)
    parsed_flow_counts = {}
    for key, value in flow_counts.items():
        parsed_flow_counts[_telemetry_key(key)] = float(value)
    parsed_window = None
    if isinstance(window, (list, tuple)) and len(window) == 2:
        parsed_window = (int(window[0]), int(window[1]))
    return {
        "flow_registry": flow_registry,
        "service_factors": service_factors,
        "pool_counts": pool_counts,
        "hbm_counts": hbm_counts,
        "telemetry_rates": parsed_rates,
        "telemetry_flow_counts": parsed_flow_counts,
        "telemetry_window": parsed_window,
    }


def _telemetry_key(key: str):
    """遥测键双形还原（"src->dst" 端点形 / 裸整型 link id）。"""
    text = str(key)
    if "->" in text:
        source, destination = text.split("->", 1)
        return (int(source), int(destination))
    return int(text)


def hydrate_session_view(stage_input: Mapping[str, Any]):
    from joint.joint_cost_model import SessionKVView
    fields = stage_input["session_view"]
    return SessionKVView(
        session_id=str(fields["session_id"]),
        home_instance=fields["home_instance"],
        resident_instance=fields["resident_instance"],
        location=str(fields["location"]),
        history_tokens=int(fields["history_tokens"]),
        resident_prefix_layers=int(fields["resident_prefix_layers"]),
        history_bytes_by_tp_rank=tuple(
            int(value) for value in fields["history_bytes_by_tp_rank"]),
        missing_bytes_by_tp_rank=tuple(
            int(value) for value in fields["missing_bytes_by_tp_rank"]),
    )


def hydrate_request_view(stage_input: Mapping[str, Any]):
    from joint.joint_cost_model import RequestView
    fields = stage_input["request_view"]
    return RequestView(
        request_id=str(fields["request_id"]),
        session_id=str(fields["session_id"]),
        input_tokens=int(fields["input_tokens"]),
        history_tokens_before=int(fields["history_tokens_before"]),
        estimated_decode_tokens=int(fields["estimated_decode_tokens"]),
        horizon_source=str(fields["horizon_source"]),
        input_kv_bytes_by_tp_rank=tuple(
            int(value) for value in fields["input_kv_bytes_by_tp_rank"]),
        prefill_scan_passes=fields.get("prefill_scan_passes"),
        prefill_chunk_tokens=fields.get("prefill_chunk_tokens"),
        query_tile_tokens=fields.get("query_tile_tokens"),
    )


def hydrate_boundary_view(stage_input: Mapping[str, Any]):
    from joint.joint_cost_model import PDBoundaryView
    fields = stage_input["boundary_view"]
    return PDBoundaryView(
        session_id=str(fields["session_id"]),
        home_instance=fields["home_instance"],
        p_instance_index=int(fields["p_instance_index"]),
        p_action=str(fields["p_action"]),
        history_tokens=int(fields["history_tokens"]),
        working_bytes_by_tp_rank=tuple(
            int(value) for value in fields["working_bytes_by_tp_rank"]),
        base_bytes_by_tp_rank=tuple(
            int(value) for value in fields["base_bytes_by_tp_rank"]),
        decode_steps=int(fields["decode_steps"]),
        base_tokens=int(fields.get("base_tokens", 0)),
        kv_token_bytes=fields.get("kv_token_bytes"),
        logic_home_instance=fields.get("logic_home_instance"),
        kv_token_bytes_by_tp_rank=(
            None if fields.get("kv_token_bytes_by_tp_rank") is None
            else tuple(int(value) for value in
                      fields["kv_token_bytes_by_tp_rank"])),
        working_intervals=tuple(
            tuple(interval) for interval in fields.get(
                "working_intervals", ())),
    )


def build_acceptance_cost_model(
        runtime: HydratedRuntime,
        planner_state: Mapping[str, Any],
        *,
        prediction_config,
        native_library,
        world_source=None,
        obligation_resolver=None,
) -> Any:
    """按 sh30 ``_joint_cost_model`` 同一注入闭包配方重建 JCM。

    全部注入闭包（路由/工作量剖面/decode 模板）都以原语函数
    （deterministic_xy_route / estimate_prefill_workload /
    estimate_iteration_workload / estimate_decode_remaining_task_load_ns）
    从重建的 hardware/model 派生，与生产装配同一来源；不写第二套公式。
    """
    from face_scheduler import (
        deterministic_xy_route,
        estimate_decode_remaining_task_load_ns,
        estimate_iteration_workload,
        estimate_prefill_task_load_ns,
        estimate_prefill_workload,
        kv_cache_shard_bytes_for_tokens,
    )
    from joint.joint_cost_model import JointCostModel
    from online.sh30_online_scheduler import _PoolPortRegistry

    stage_input = planner_state["stage_input"]
    hydrated = hydrate_registries(planner_state)
    loads = hydrate_instance_loads(planner_state)
    if not loads:
        raise AcceptanceProjectionIncomplete(
            "planner_state.instance_loads is empty; cannot rebuild the "
            "production candidate face")
    hardware = runtime.hardware
    model = runtime.model
    topology = runtime.topology
    instance_size = topology.instances[0].size

    def route(source_instance: int, target_instance: int):
        source_group = topology.instance(source_instance)
        target_group = topology.instance(target_instance)
        source_rank = source_group.ranks[0]
        target_rank = target_group.ranks[0]
        path = deterministic_xy_route(hardware, source_rank, target_rank)
        return path, max(0, len(path) - 1)

    def route_paths(source_instance: int, target_instance: int):
        source_group = topology.instance(source_instance)
        target_group = topology.instance(target_instance)
        return [
            deterministic_xy_route(hardware, source_rank, target_rank)
            for source_rank, target_rank in zip(
                source_group.ranks, target_group.ranks)]

    def rank_route(source_rank: int, target_rank: int):
        return deterministic_xy_route(
            hardware, source_rank, target_rank)

    prefill_ns_per_token = float(estimate_prefill_task_load_ns(
        hardware, model, instance_size=instance_size,
        chunk_tokens=1, context_tokens=1))
    decode_ns_per_token = float(
        estimate_decode_remaining_task_load_ns(
            hardware, model, instance_size=instance_size,
            current_context_tokens=1, generated_tokens=0,
            average_decode_length=1.0,
            running_step_fraction_remaining=1.0,
            hbm_share_divisor=1))

    def decode_task_load_ns(context_tokens: int, *,
                            hbm_share_divisor: int = 1) -> int:
        return estimate_decode_remaining_task_load_ns(
            hardware, model, instance_size=instance_size,
            current_context_tokens=int(context_tokens),
            generated_tokens=0,
            average_decode_length=1.0,
            running_step_fraction_remaining=1.0,
            hbm_share_divisor=hbm_share_divisor)

    p_chunk = runtime.p_chunk

    def prefill_work_profile(input_tokens: int,
                             history_tokens: int) -> tuple[int, int]:
        profile = estimate_prefill_workload(
            hardware, model, instance_size=instance_size,
            input_tokens=int(input_tokens),
            history_tokens=int(history_tokens),
            p_chunk=p_chunk)
        return profile.pure_compute_ns, profile.hbm_bytes_per_rank

    def decode_work_profile(context_tokens: int) -> tuple[int, int]:
        profile = estimate_iteration_workload(
            hardware, model, instance_size=instance_size,
            p_chunk=0, d_batch=1, d_token=int(context_tokens),
            p_context_tokens=0)
        return profile.pure_compute_ns, profile.hbm_bytes_per_rank

    def prefill_pure_compute_ns(input_tokens, history_tokens):
        return prefill_work_profile(input_tokens, history_tokens)[0]

    def prefill_hbm_bytes(input_tokens, history_tokens):
        return prefill_work_profile(input_tokens, history_tokens)[1]

    def prefill_latency_profile(input_tokens, history_tokens):
        profile = estimate_prefill_workload(
            hardware, model, instance_size=instance_size,
            input_tokens=int(input_tokens),
            history_tokens=int(history_tokens),
            p_chunk=p_chunk)
        return profile.hbm_latency_ns, profile.controller_latency_ns

    def decode_pure_compute_ns(context_tokens):
        return decode_work_profile(context_tokens)[0]

    def decode_hbm_bytes(context_tokens):
        return decode_work_profile(context_tokens)[1]

    def decode_latency_profile(context_tokens):
        profile = estimate_iteration_workload(
            hardware, model, instance_size=instance_size,
            p_chunk=0, d_batch=1, d_token=int(context_tokens),
            p_context_tokens=0)
        return profile.hbm_latency_ns, profile.controller_latency_ns

    # 池端口/HBM 端口注册表：原类 + 原登记原语重建计数。
    pool_ports = _PoolPortRegistry()
    for edge_rank in sorted(hydrated["pool_counts"]):
        for _ in range(hydrated["pool_counts"][edge_rank]):
            pool_ports.register(int(edge_rank), owner="hydrated#acceptance")

    from joint.hbm_port_flow_registry import HbmPortFlowRegistry
    hbm_ports = HbmPortFlowRegistry()
    hbm_counts = hydrated["hbm_counts"]

    def active_decode_provider(rank: int) -> int:
        return hbm_counts.get(int(rank), (0, 0))[1]

    hbm_ports.attach_active_decode_provider(active_decode_provider)
    for rank in sorted(hbm_counts):
        for _ in range(hbm_counts[rank][0]):
            hbm_ports.register(
                int(rank), owner="hydrated#acceptance")

    telemetry_kwargs = {}
    rates_map = hydrated["telemetry_rates"]
    if rates_map:
        telemetry_kwargs["link_telemetry_rates"] = dict(rates_map)
    flow_counts_map = hydrated["telemetry_flow_counts"]
    if flow_counts_map:
        telemetry_kwargs["link_telemetry_flow_counts"] = dict(
            flow_counts_map)
    window = hydrated["telemetry_window"]
    if window is not None:
        telemetry_kwargs["link_telemetry_window_ns"] = tuple(window)

    if "prefill_task_load_ns_fn" in JointCostModel.__dataclass_fields__:
        telemetry_kwargs["prefill_task_load_ns_fn"] = (
            _joint_prefill_total_load_ns_equivalent(
                hardware, model, instance_size, p_chunk))
    if "decode_task_load_ns_fn" in JointCostModel.__dataclass_fields__:
        telemetry_kwargs["decode_task_load_ns_fn"] = decode_task_load_ns
    if "prefill_pure_compute_ns_fn" in JointCostModel.__dataclass_fields__:
        telemetry_kwargs["prefill_pure_compute_ns_fn"] = (
            prefill_pure_compute_ns)
    if "prefill_hbm_bytes_fn" in JointCostModel.__dataclass_fields__:
        telemetry_kwargs["prefill_hbm_bytes_fn"] = prefill_hbm_bytes
    if "decode_pure_compute_ns_fn" in JointCostModel.__dataclass_fields__:
        telemetry_kwargs["decode_pure_compute_ns_fn"] = (
            decode_pure_compute_ns)
    if "decode_hbm_bytes_fn" in JointCostModel.__dataclass_fields__:
        telemetry_kwargs["decode_hbm_bytes_fn"] = decode_hbm_bytes
    if "prefill_latency_profile_fn" in JointCostModel.__dataclass_fields__:
        telemetry_kwargs["prefill_latency_profile_fn"] = (
            prefill_latency_profile)
    if "decode_latency_profile_fn" in JointCostModel.__dataclass_fields__:
        telemetry_kwargs["decode_latency_profile_fn"] = (
            decode_latency_profile)

    return JointCostModel(
        rates=runtime.rates,
        loads=loads,
        flow_registry=hydrated["flow_registry"],
        service_factors=hydrated["service_factors"],
        prefill_ns_per_token=prefill_ns_per_token,
        decode_ns_per_token=decode_ns_per_token,
        model_layers=model.layers,
        instance_tp_size=instance_size,
        route_fn=route,
        route_paths_fn=route_paths,
        nearest_edge_fn=runtime.kv_manager.nearest_edge,
        rank_route_fn=rank_route,
        instance_ranks_fn=lambda index: tuple(
            topology.instance(index).ranks),
        pool_edge_flows_fn=pool_ports.count,
        hbm_port_registry=hbm_ports,
        remote_read_tile_bytes=runtime.joint_config.remote_read_tile_bytes,
        remote_read_kv_token_bytes=(
            kv_cache_shard_bytes_for_tokens(model, 1, instance_size)[0]),
        remote_read_partial=(
            runtime.joint_config.remote_read_partial_enabled),
        allowed_actions=runtime.joint_config.allowed_action_set,
        prediction_config=prediction_config,
        native_projection_library=native_library,
        **({"world_source": world_source}
           if world_source is not None else {}),
        **({"obligation_resolver": obligation_resolver}
           if obligation_resolver is not None else {}),
        **telemetry_kwargs,
    )


def _joint_prefill_total_load_ns_equivalent(hardware, model, instance_size,
                                            p_chunk):
    """prefill 整段负载闭包——与 sh30 ``_joint_prefill_total_load_ns``
    同一原语配方（``estimate_prefill_task_load_ns`` + ``fold_repeat_
    service_ns`` K=min(8,m) 区段中点折叠；memo 仅为缓存，不入配方）。"""

    def prefill_total_load_ns(input_tokens: int, history_tokens: int, *,
                              hbm_share_divisor: int = 1) -> int:
        from face_scheduler import estimate_prefill_task_load_ns
        from joint.joint_cost_model import fold_repeat_service_ns

        def chunk_load(chunk_tokens: int, context_tokens: int) -> int:
            return int(estimate_prefill_task_load_ns(
                hardware, model, instance_size=instance_size,
                chunk_tokens=int(chunk_tokens),
                context_tokens=int(context_tokens),
                hbm_share_divisor=int(hbm_share_divisor)))

        input_tokens = int(input_tokens)
        if input_tokens <= 0:
            return 0
        chunk_tokens_full = min(int(p_chunk), input_tokens)
        if chunk_tokens_full <= 0:
            raise ValueError("prefill chunk size must be positive")
        full_chunks, tail_tokens = divmod(input_tokens, chunk_tokens_full)
        total = 0.0
        if full_chunks >= 1:
            total += chunk_load(
                chunk_tokens_full,
                history_tokens + chunk_tokens_full)
            remaining_chunks = full_chunks - 1
            if remaining_chunks > 0:
                total += fold_repeat_service_ns(
                    steps=remaining_chunks,
                    first_context_tokens=(
                        history_tokens + 2 * chunk_tokens_full),
                    context_stride_tokens=chunk_tokens_full,
                    service_fn=chunk_tokens_of(chunk_load,
                                               chunk_tokens_full))
        if tail_tokens > 0:
            total += chunk_load(tail_tokens, history_tokens + input_tokens)
        return int(total)

    def chunk_tokens_of(chunk_load, chunk_tokens_full):
        def service_fn(context_tokens: int) -> int:
            return chunk_load(chunk_tokens_full, int(context_tokens))
        return service_fn

    return prefill_total_load_ns




# ====================================================== evaluate_boundary ==


def _failure_dto(reason: str, *, boundary: Mapping[str, Any] | None,
                 status: str) -> dict:
    """最小 failure DTO（只 version/status/reason + 已知身份）。"""
    dto = {"version": DTO_VERSION, "status": status, "reason": str(reason)}
    if boundary is not None:
        for key in ("request_id", "stage", "boundary_ordinal", "tick_ns"):
            value = boundary.get(key)
            if value is not None:
                dto[key] = value
    return dto


def _candidate_rows(candidates, *, applicable_only: bool) -> list:
    """候选行（candidates / inapplicable_candidates 的分面行）。"""
    from joint.joint_scheduler import _candidate_id
    rows = []
    for candidate in candidates:
        row = {
            "candidate_id": _candidate_id(candidate),
            "instance_id": int(candidate.instance_index),
            "action": candidate.action,
        }
        if applicable_only:
            from joint.native_projection import _work_wire
            work = getattr(getattr(candidate, "projection_graph", None),
                           "work", None)
            row["work_item"] = (
                _work_wire(work) if work is not None else None)
        else:
            row["reason"] = candidate.inapplicable_reason
        rows.append(row)
    return rows


def _result_rows(evaluation_rows) -> list:
    """approx/exact results 行（candidate_id/status/service_ns/settle_ns/
    quality/result；result 保留完整原评估行）。"""
    rows = []
    for row in evaluation_rows or ():
        rows.append({
            "candidate_id": row["candidate_id"],
            "status": row["status"],
            "service_ns": row["service_ns"],
            "settle_ns": row["settle_ns"],
            "quality": dict(row.get("native_kernel") or {}),
            "result": dict(row),
        })
    return rows


def _load_library(library_path):
    from joint.native_projection import NativeProjectionLibrary
    return NativeProjectionLibrary(library_path)


def _hydrate_world_source(boundary: Mapping[str, Any],
                          runtime: HydratedRuntime):
    """从 causal_prestate.work_ledger 重建背景世界源。

    certified_empty → None（JCM 自构 LoadsWorldSource，保留降级披露）；
    replay_v3_frames 保留旧字段名，但内容是带版本身份的 retained raw
    frame 序列；按首帧严格选择 V3 archive 或 V4 offline 双账本 decoder，
    整序列必须同版。非空但无真实重放输入的状态明确 INCOMPLETE，不伪造背景。
    """
    prestate = boundary.get("causal_prestate") or {}
    work_ledger = prestate.get("work_ledger") or {}
    if work_ledger.get("certified_empty", True):
        return None
    prebuilt = work_ledger.get("_prebuilt_ledger_world_source")
    if prebuilt is not None:
        # 批量 L3 优化（2026-10-07）：调用方（l3-batch 工具）逐案例增量
        # 重放并为每个边界预构 LedgerWorldSource——避免逐边界全前缀重放
        # 的 O(N^2)（53 边界案例实测单边界 >4s）。语义等价：预构对象
        # 同样来自生产入口 _apply_work_ledger 的累积态。
        return prebuilt
    frames = work_ledger.get("replay_v3_frames")
    if not isinstance(frames, list) or not frames:
        raise AcceptanceProjectionIncomplete(
            "causal_prestate.work_ledger is non-empty but carries no "
            "replayable raw v3 archive or v4 retained frames; "
            "refusing to fabricate a ledger "
            "background (provide replay_v3_frames or an empty certified "
            "state)")
    protocol_version = _retained_frame_protocol_version(frames)
    harness = _build_dual_ledger_harness(protocol_version)
    for frame in frames:
        harness._apply_work_ledger({"work_ledger": frame})
    from joint.runtime_snapshot import build_decision_snapshot
    tick_ns = int(boundary.get("tick_ns", 0))
    delta = {
        "delivery_sequence": int(harness.joint_pipeline_ledger.delivery_sequence),
        "tick": tick_ns,
        "arrivals": [],
    }
    snapshot = build_decision_snapshot(
        delta,
        versions=None,
        ledger_state={},
        ledger_projection=harness.joint_work_ledger.pricing_snapshot(),
        pipeline_ledger_state=harness.joint_pipeline_ledger.pricing_snapshot(),
        pricing_only=True)
    from joint.action_work_graph import LedgerWorldSource
    from online.sh30_online_scheduler import Sh30OnlineScheduler
    # Reuse the production mesh LinkId enumeration from the verified runtime
    # asset so the restored authoritative NoC frontier is priced as well.
    link_mapper = Sh30OnlineScheduler.__new__(Sh30OnlineScheduler)
    link_mapper.hardware = runtime.hardware
    link_mapper._telemetry_link_id_map = None
    return LedgerWorldSource(
        ledger=harness.joint_work_ledger, snapshot=snapshot,
        link_id_map_fn=link_mapper._telemetry_endpoint_link_key)


def _resolve_runtime_asset(boundary: Mapping[str, Any],
                           fixture_state: Mapping[str, Any] | None,
                           private_dir: Path) -> HydratedRuntime:
    """fixture_state → runtime asset（嵌入 dict 或已验摘要文件）。"""
    if not isinstance(fixture_state, dict):
        raise AcceptanceProjectionIncomplete(
            "fixture_state is required (assets + python_runtime_asset_id)")
    asset_id = fixture_state.get("python_runtime_asset_id")
    if asset_id is None:
        raise AcceptanceProjectionIncomplete(
            "fixture_state.python_runtime_asset_id is required")
    assets = fixture_state.get("assets")
    if isinstance(assets, list):
        # Driver fixture-state schema: assets is a list of
        # {asset_id, path, sha256} with paths relative to the state file
        # (the validated state carries _verified_asset_root as an
        # attribute). Resolve with byte verification, then hydrate.
        asset = _resolve_runtime_asset_strict(fixture_state)
        return hydrate_runtime_asset(asset, private_dir=private_dir)
    assets = assets or {}
    entry = assets.get(asset_id)
    if isinstance(entry, Mapping) and "face_trace" in entry:
        asset = dict(entry)
    elif isinstance(entry, Mapping) and entry.get("path"):
        path = Path(entry["path"])
        if not path.is_absolute():
            root = fixture_state.get("root")
            if not root:
                raise AcceptanceProjectionIncomplete(
                    "relative asset path requires fixture_state.root")
            path = Path(root) / path
        if not path.is_file():
            raise AcceptanceProjectionIncomplete(
                f"runtime asset file {str(path)!r} does not exist")
        expected = entry.get("sha256")
        actual = _sha256_file(path)
        if expected is not None and actual != expected:
            raise AcceptanceProjectionFailure(
                f"runtime asset sha256 mismatch for {str(path)!r}: "
                f"expected {expected}, got {actual}")
        asset = json.loads(path.read_text(encoding="utf-8"))
    elif asset_id == "inline:planner_state_runtime":
        # 有界证明路径：runtime asset 内嵌在 fixture_state（同源导出）。
        asset = fixture_state.get("python_runtime_asset")
        if not isinstance(asset, Mapping):
            raise AcceptanceProjectionIncomplete(
                "fixture_state.python_runtime_asset is required when "
                "python_runtime_asset_id == 'inline:planner_state_runtime'")
    else:
        raise AcceptanceProjectionIncomplete(
            f"fixture_state.assets[{asset_id!r}] is missing or malformed")
    return hydrate_runtime_asset(asset, private_dir=private_dir)


def evaluate_boundary(boundary, prediction_config, *, library_path,
                      fixture_state) -> dict:
    """L3 纯验收入口（§19.1）。

    从真实 boundary 的 causal_prestate/metadata + planner_state 用原
    构造器重建原生产 snapshot/planner，构造全部适用候选，走原
    ``select_instance_and_action`` / ``select_decode_target_and_action``
    与原 JCM/native 投影；返回 ``projection-evaluation-v1-20261006``
    DTO。不写第二套估价/排名。
    """
    import tempfile

    from joint.joint_config import MODEL_SEMANTICS_VERSION
    from joint.prediction_config import (
        make_pipeline_kernel_contract,
        require_prediction_config,
    )
    from joint import joint_scheduler

    if not isinstance(boundary, dict):
        raise AcceptanceProjectionIncomplete(
            "boundary must be a JSON object")
    try:
        _require_keys(
            boundary,
            ("request_id", "stage", "boundary_ordinal", "tick_ns",
             "causal_prestate", "causal_metadata", "planner_state"),
            where="boundary")
        stage = boundary["stage"]
        if stage not in _STAGE_VALUES:
            raise AcceptanceProjectionIncomplete(
                f"boundary.stage must be one of {_STAGE_VALUES}")
        planner_state = boundary["planner_state"]
        _require_keys(
            planner_state, ("version", "stage_input", "instance_loads",
                            "registries", "forecast_generator"),
            where="planner_state")
        if planner_state["version"] != PLANNER_STATE_VERSION:
            raise AcceptanceProjectionIncomplete(
                f"planner_state.version {planner_state['version']!r} != "
                f"{PLANNER_STATE_VERSION!r}")
        prediction_config = require_prediction_config(prediction_config)
        if library_path is None:
            raise AcceptanceProjectionIncomplete(
                "library_path (same-ABI native projection library) is "
                "required")
        library_path = Path(library_path)
        if not library_path.is_file():
            raise AcceptanceProjectionIncomplete(
                f"native projection library {str(library_path)!r} does "
                "not exist")
        library_sha256 = _sha256_file(library_path)

        metadata = boundary["causal_metadata"]
        _require_keys(
            metadata, ("context_tokens", "horizon_tokens",
                       "forecast_generator"),
            where="causal_metadata")
        horizon_tokens = int(metadata["horizon_tokens"])
        if horizon_tokens <= 0:
            raise AcceptanceProjectionIncomplete(
                "causal_metadata.horizon_tokens must be positive")
        generator = metadata["forecast_generator"]
        _require_keys(
            generator, ("identity", "causal_inputs", "semantic_sha256"),
            where="causal_metadata.forecast_generator")
        causal_inputs = generator["causal_inputs"]
        _require_keys(
            causal_inputs, ("context_tokens", "horizon_tokens",
                            "generator_inputs"),
            where="forecast_generator.causal_inputs")
        if int(causal_inputs["horizon_tokens"]) != horizon_tokens:
            raise AcceptanceProjectionIncomplete(
                "forecast horizon_tokens disagrees with causal_metadata")

        with tempfile.TemporaryDirectory(
                prefix="joint-acceptance-hydrate-") as private_dir:
            runtime = _resolve_runtime_asset(
                boundary, fixture_state, Path(private_dir))
            library = _load_library(library_path)
            world_source = _hydrate_world_source(boundary, runtime)

            stage_input = planner_state["stage_input"]
            mode = runtime.joint_config.scheduler_mode
            cost_model = build_acceptance_cost_model(
                runtime, planner_state,
                prediction_config=prediction_config,
                native_library=library,
                world_source=world_source)
            if stage_input.get("kind") != stage:
                raise AcceptanceProjectionIncomplete(
                    "planner_state.stage_input.kind disagrees with "
                    "boundary.stage")
            if stage == "P":
                session = hydrate_session_view(stage_input)
                request = hydrate_request_view(stage_input)
                if request.estimated_decode_tokens != horizon_tokens:
                    raise AcceptanceProjectionIncomplete(
                        "request_view.estimated_decode_tokens disagrees "
                        "with causal horizon_tokens")
                prep = {
                    int(index): int(value) for index, value in
                    stage_input.get("prep_frontier_by_instance",
                                    {}).items()}
                record = joint_scheduler.select_instance_and_action(
                    mode=mode,
                    cost_model=cost_model,
                    session=session,
                    request=request,
                    remote_enabled=bool(
                        stage_input.get("remote_enabled", True)),
                    prep_frontier_by_instance=prep)
            else:
                boundary_view = hydrate_boundary_view(stage_input)
                if boundary_view.decode_steps != horizon_tokens:
                    raise AcceptanceProjectionIncomplete(
                        "boundary_view.decode_steps disagrees with "
                        "causal horizon_tokens")
                record = joint_scheduler.select_decode_target_and_action(
                    mode=mode,
                    cost_model=cost_model,
                    boundary=boundary_view,
                    remote_enabled=bool(
                        stage_input.get("remote_enabled", True)),
                    decision_epoch=int(
                        stage_input.get("decision_epoch", 0)),
                    inherited_obligation_ids=tuple(
                        stage_input.get("inherited_obligation_ids", ())))
    except AcceptanceProjectionIncomplete as exc:
        return _failure_dto(
            str(exc), boundary=boundary, status="INCOMPLETE")
    except Exception as exc:  # 真实 backend/预算失败 → FAIL（保原始证据）。
        return _failure_dto(
            f"{type(exc).__name__}: {exc}", boundary=boundary,
            status="FAIL")

    arbitration = record.diagnostics.get("projection_arbitration")
    if arbitration is None:
        raise AcceptanceProjectionFailure(
            "selection record lacks projection_arbitration diagnostics")

    applicable = [
        candidate for candidate in record.candidates
        if candidate.applicable and candidate.cost_ns is not None]
    inapplicable = [
        candidate for candidate in record.candidates
        if not (candidate.applicable and candidate.cost_ns is not None)]

    # projection_input：选定候选的原 typed ProjectionInput wire（同一次
    # 冻结世界上下文）。
    from joint.native_projection import projection_input_to_wire
    chosen_graph = getattr(record.chosen, "projection_graph", None)
    chosen_world = getattr(record.chosen, "projection_world", None)
    if chosen_graph is None or chosen_world is None:
        raise AcceptanceProjectionFailure(
            "chosen candidate lacks its retained projection graph/world")
    projection_input = projection_input_to_wire(
        chosen_world.projection_input(chosen_graph))

    config_sha = make_pipeline_kernel_contract(prediction_config)[
        "prediction_config_sha256"]
    dto = {
        "version": DTO_VERSION,
        "status": "PASS",
        "reason": None,
        "request_id": boundary["request_id"],
        "stage": stage,
        "boundary_ordinal": int(boundary["boundary_ordinal"]),
        "tick_ns": int(boundary["tick_ns"]),
        "horizon_tokens": horizon_tokens,
        "model_semantics_version": MODEL_SEMANTICS_VERSION,
        "projection_input": projection_input,
        "candidates": _candidate_rows(applicable, applicable_only=True),
        "inapplicable_candidates": _candidate_rows(
            inapplicable, applicable_only=False),
        "forecast_generator": {
            "identity": generator["identity"],
            "causal_inputs": {
                "context_tokens": int(
                    causal_inputs["context_tokens"]),
                "horizon_tokens": int(
                    causal_inputs["horizon_tokens"]),
                "generator_inputs": dict(
                    causal_inputs["generator_inputs"]),
            },
            "semantic_sha256": generator["semantic_sha256"],
        },
        "approx_results": _result_rows(arbitration["approx_results"]),
        "exact_results": _result_rows(arbitration["exact_results"]),
        "arbitration": {
            "trigger_reason": arbitration.get("trigger_reason"),
            "exact_recomputed_ids": list(
                arbitration.get("exact_recomputed_ids") or ()),
            "selected_id": arbitration.get("selected_id"),
            "approximate_selected_id": arbitration.get(
                "approximate_selected_id"),
            "exact_selected_id": arbitration.get("exact_selected_id"),
            "old_tie_order": list(
                arbitration.get("old_tie_order") or ()),
        },
        "library_sha256": library_sha256,
        "prediction_config_sha256": config_sha,
        "world_source_status": (
            "ledger" if world_source is not None
            else "absent_legacy_peer"),
    }
    _verify_dto_consistency(dto)
    return dto


def _verify_dto_consistency(dto: dict) -> None:
    """PASS DTO 的互斥/覆盖一致性（不造 winner）。"""
    applicable_ids = [row["candidate_id"] for row in dto["candidates"]]
    inapplicable_ids = [
        row["candidate_id"] for row in dto["inapplicable_candidates"]]
    if len(set(applicable_ids)) != len(applicable_ids):
        raise AcceptanceProjectionFailure(
            "duplicate applicable candidate ids in PASS DTO")
    if set(applicable_ids) & set(inapplicable_ids):
        raise AcceptanceProjectionFailure(
            "applicable/inapplicable candidate ids overlap")
    if dto["arbitration"]["selected_id"] not in applicable_ids:
        raise AcceptanceProjectionFailure(
            "arbitration.selected_id is outside the applicable face")
    approx_winner = dto["arbitration"]["approximate_selected_id"]
    if approx_winner is None or approx_winner == "invalid":
        raise AcceptanceProjectionFailure(
            "no valid finite approximate winner (invalid; no winner "
            "fabricated)")
    if approx_winner not in applicable_ids:
        raise AcceptanceProjectionFailure(
            "approximate winner is outside the applicable face")
    exact_winner = dto["arbitration"]["exact_selected_id"]
    if exact_winner is not None and exact_winner not in applicable_ids:
        raise AcceptanceProjectionFailure(
            "exact winner is outside the applicable face")


# ============================================== Stage B canonicalization ==

#: §19.1A Stage A audit-record discriminator and exact entry keys. The
#: discriminator is retained for the frozen fixture wire shape; ``frame``
#: carries the exact V3 archive or current V4 sparse-patch envelope.
RAW_FRAME_AUDIT_KIND = "raw_v2_frame"
RAW_FRAME_AUDIT_ENTRY_KEYS = frozenset({
    "kind", "boundary_ordinal", "frame", "runtime_snapshot",
    "producer_build"})


class _OfflineDualLedgerHarness:
    """Explicit offline replay for exact archived v3 or current v4 frames.

    v4（当前线上格式，joint 加速改造 v4 起）经同一原生双账本接收器重放
    （joint.native_ledger.NativeDualLedgerReceiver——与在线生产完全相同
    的语义核心，不存在第二套线上状态机）。v3 是冻结的历史档档格式：
    保留 Python 参考解码器（PipelineLedger）做边界读取（历史档案对照
    专用，非线上状态机）。
    """

    def __init__(self, protocol_version):
        if protocol_version == 4:
            from joint.native_ledger import (
                NativeDualLedgerReceiver,
                NativePipelineLedgerFacade,
            )
            self._native_receiver = NativeDualLedgerReceiver()
            self.joint_work_ledger = self._native_receiver
            self.joint_pipeline_ledger = NativePipelineLedgerFacade(
                self._native_receiver)
        elif protocol_version == 3:
            from joint.pipeline_ledger import PipelineLedger
            from joint.work_ledger import WorkLedger

            self._native_receiver = None
            self.joint_work_ledger = WorkLedger()
            self.joint_pipeline_ledger = PipelineLedger()
        else:
            raise AcceptanceProjectionIncomplete(
                f"offline pipeline protocol {protocol_version!r} is unsupported")
        self.work_ledger_absent_seen = False

    def _apply_work_ledger(self, delta):
        payload = delta.get("work_ledger")
        if self._native_receiver is not None:
            receipt = self._native_receiver.receive_payload_dict(payload)
            return json.dumps(
                payload, sort_keys=True, separators=(",", ":"),
                ensure_ascii=False, allow_nan=False).encode("utf-8")
        parsed = self.joint_pipeline_ledger.validate_envelope(payload)
        work_tx = self.joint_work_ledger.begin_protocol_v3_transaction(
            delivery_sequence=payload.get("delivery_sequence"),
            tick=payload.get("tick"), snapshot_kind=payload.get("snapshot_kind"))
        prepared = self.joint_pipeline_ledger.prepare_envelope(
            payload, work_transaction=work_tx, _parsed=parsed)
        work_tx.commit(notify=False)
        self.joint_pipeline_ledger.commit_prepared(prepared)
        work_tx.notify()
        return prepared.parsed.canonical_bytes


def _build_dual_ledger_harness(protocol_version):
    """Choose one strict decoder for an offline retained-frame sequence."""
    return _OfflineDualLedgerHarness(protocol_version)


def _retained_frame_protocol_version(frames):
    """Select one strict offline decoder from a V3/V4 retained sequence.

    V3 remains an archive decoder and V4 is the current offline frame shape.
    This selector is never used by the online scheduler, which is V4-only.
    """
    if not isinstance(frames, (list, tuple)):
        raise AcceptanceProjectionIncomplete(
            "offline retained frames must be an ordered array")
    if not frames:
        raise AcceptanceProjectionIncomplete(
            "offline retained-frame sequence is empty; no protocol can be selected")
    first = frames[0]
    if not isinstance(first, Mapping):
        raise AcceptanceProjectionIncomplete(
            "offline retained frame[0] must be an object")
    protocol_version = first.get("protocol_version")
    if (type(protocol_version) is not int
            or protocol_version not in (3, 4)
            or type(first.get("schema_version")) is not int
            or first.get("schema_version") != protocol_version):
        raise AcceptanceProjectionIncomplete(
            "offline retained frame[0] must identify exact protocol 3 archive "
            "or protocol 4")
    for index, frame in enumerate(frames):
        if not isinstance(frame, Mapping):
            raise AcceptanceProjectionIncomplete(
                f"offline retained frame[{index}] must be an object")
        schema_version = frame.get("schema_version")
        version = frame.get("protocol_version")
        if (type(schema_version) is not int or type(version) is not int
                or schema_version != protocol_version
                or version != protocol_version):
            raise AcceptanceProjectionIncomplete(
                "offline retained frame sequence mixes V3 archive and V4 "
                "protocols or has a malformed version header")
    return protocol_version


def _replay_ledger_views(frames):
    """按边界序用显式 v3 archive 或 v4 offline decoder 重放 raw frames。"""
    protocol_version = _retained_frame_protocol_version(frames)
    harness = _build_dual_ledger_harness(protocol_version)
    for frame in frames:
        harness._apply_work_ledger({"work_ledger": frame})
    ordinary = harness.joint_work_ledger.export_state()
    pipeline = _jsonify(harness.joint_pipeline_ledger.remaining_view())
    return {
        "certified_empty": False,
        "ordinary": ordinary,
        "pipeline": pipeline,
        "replayed_frame_count": len(frames),
        "work_ledger_version": int(harness.joint_work_ledger.version),
        "work_ledger_watermark": int(
            harness.joint_work_ledger.event_watermark),
    }


def _collect_raw_v2_frames(audit_records):
    """从 audit_records **数组**过滤 kind=="raw_v2_frame" 条目（与驱动
    validate_raw_fixture_result 完全同规则；非 raw 条目跳过）。"""
    if not isinstance(audit_records, list):
        raise AcceptanceProjectionIncomplete(
            "raw fixture result audit_records must be an array")
    frames = []
    for index, entry in enumerate(audit_records):
        if not isinstance(entry, dict):
            raise AcceptanceProjectionIncomplete(
                f"audit_records[{index}] must be a JSON object")
        if entry.get("kind") != RAW_FRAME_AUDIT_KIND:
            continue
        if set(entry) != set(RAW_FRAME_AUDIT_ENTRY_KEYS):
            raise AcceptanceProjectionIncomplete(
                f"audit_records[{index}] raw_v2_frame entry keys must be "
                f"exactly {sorted(RAW_FRAME_AUDIT_ENTRY_KEYS)}")
        for field in ("frame", "runtime_snapshot", "producer_build"):
            if not isinstance(entry[field], dict):
                raise AcceptanceProjectionIncomplete(
                    f"audit_records[{index}].{field} must be a JSON object")
        frame = entry["frame"]
        schema_version = frame.get("schema_version")
        protocol_version = frame.get("protocol_version")
        if (type(schema_version) is not int
                or type(protocol_version) is not int
                or (schema_version, protocol_version) not in ((3, 3), (4, 4))
                or (schema_version == 3 and (
                    frame.get("bridge_protocol") != "bridge-v3-20261009"
                    or frame.get("work_ledger_schema")
                        != "workledger-v3-20261009"))
                or (schema_version == 4 and (
                    frame.get("bridge_protocol") != "bridge-v4-20261011"
                    or frame.get("work_ledger_schema")
                        != "workledger-v4-20261011"))):
            raise AcceptanceProjectionIncomplete(
                f"audit_records[{index}].frame must carry a supported exact "
                "v3 archive or v4 work-ledger envelope")
        if frames and frames[0]["frame"]["protocol_version"] != protocol_version:
            raise AcceptanceProjectionIncomplete(
                "raw frame sequence cannot mix v3 archive and v4 live envelopes")
        frames.append(entry)
    return frames


def _pair_frames_in_commit_order(frames, boundary_records):
    """raw 帧与 boundary_records 按提交顺序 1:1 配对；同位序校验
    boundary_ordinal 相等（驱动 validate_raw_fixture_result 同规则——
    P/D 可同 ordinal，判据是提交位置而非 ordinal 检索）。"""
    if len(frames) != len(boundary_records):
        raise AcceptanceProjectionIncomplete(
            f"raw v2 frames ({len(frames)}) must pair 1:1 in commit order "
            f"with boundary records ({len(boundary_records)})")
    pairs = []
    for position, (entry, record) in enumerate(zip(frames, boundary_records)):
        frame_ordinal = entry.get("boundary_ordinal")
        if isinstance(frame_ordinal, bool) or not isinstance(frame_ordinal, int):
            raise AcceptanceProjectionIncomplete(
                f"raw frame {position} boundary_ordinal must be an integer")
        if frame_ordinal != int(record["boundary_ordinal"]):
            raise AcceptanceProjectionIncomplete(
                f"raw frame {position} ordinal {frame_ordinal} does not "
                f"match boundary {record['boundary_ordinal']} at the same "
                "commit position")
        pairs.append((entry, record))
    return pairs


def _resolve_runtime_asset_strict(fixture_state):
    """Stage B 资产解析：**只**经 fixture_state["assets"]（列表条目
    {asset_id, path, sha256}），path 相对状态文件目录（驱动验证态附带
    _verified_asset_root；独立调用时可给 root），逐字节核对 SHA256 后
    解析 JSON。不读任何额外 fixture_state 键，不依赖边界记录携带。"""
    if not isinstance(fixture_state, dict):
        raise AcceptanceProjectionIncomplete(
            "fixture_state is required for Stage B asset resolution")
    asset_id = fixture_state.get("python_runtime_asset_id")
    if not isinstance(asset_id, str) or not asset_id:
        raise AcceptanceProjectionIncomplete(
            "fixture_state.python_runtime_asset_id is required")
    assets = fixture_state.get("assets")
    if not isinstance(assets, list) or not assets:
        raise AcceptanceProjectionIncomplete(
            "fixture_state.assets must be a non-empty array")
    entry = None
    for item in assets:
        if isinstance(item, dict) and item.get("asset_id") == asset_id:
            entry = item
            break
    if entry is None:
        raise AcceptanceProjectionIncomplete(
            f"fixture_state.assets carries no entry for {asset_id!r}")
    relpath = entry.get("path")
    expected = entry.get("sha256")
    if not isinstance(relpath, str) or not relpath:
        raise AcceptanceProjectionIncomplete(
            "asset entry path must be a non-empty relative path")
    # 驱动验证态把资产根挂在对象属性（_ValidatedFixtureState），独立
    # 调用可给 "root" 键；两者皆缺省回退 cwd。
    base = getattr(fixture_state, "_verified_asset_root", None)
    if base is None:
        base = fixture_state.get("root")
    if base is None:
        base = os.getcwd()
    asset_path = (Path(base) / relpath).resolve()
    if Path(base).resolve() not in asset_path.parents and \
            asset_path != Path(base).resolve():
        raise AcceptanceProjectionIncomplete(
            "asset path escapes the state directory")
    if not asset_path.is_file():
        raise AcceptanceProjectionIncomplete(
            f"runtime asset file {str(asset_path)!r} does not exist")
    actual = _sha256_file(asset_path)
    if expected is not None and actual != expected:
        raise AcceptanceProjectionFailure(
            f"runtime asset sha256 mismatch for {str(asset_path)!r}: "
            f"expected {expected}, got {actual}")
    try:
        asset = json.loads(asset_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise AcceptanceProjectionIncomplete(
            f"runtime asset {str(asset_path)!r} is not parseable JSON: "
            f"{exc}") from exc
    if not isinstance(asset, dict):
        raise AcceptanceProjectionIncomplete(
            "runtime asset must be a JSON object")
    return asset


def _seed_cold_base_sessions(shell, initial_state):
    """按 initial_state.history.sessions 的已提交基布局事实，用**原构造器**
    在 hydrate 的 KVCacheManager 上重建基会话（主 agent 裁决 2026-10-06
    路线 (i)：loaded logical 冷初态承载基布局；执行真值不泄入）。

    每条事实 {session_id, home_instance, location, context_tokens,
    resident_prefix_layers}：prepare_prefill -> expand_prefill ->
    mark_complete ->（PARTIAL 时）_evict_suffix——与生产者播种完全同源
    的原语序列；零事实 = 真实无基冷首（不伪造）。"""
    history = initial_state.get("history") if isinstance(
        initial_state, dict) else None
    sessions = (
        history.get("sessions") if isinstance(history, dict) else None)
    if not isinstance(sessions, list):
        return
    manager = shell.kv_manager
    for fact in sessions:
        if not isinstance(fact, dict):
            continue
        session_id = str(fact.get("session_id") or "")
        home = fact.get("home_instance")
        context = int(fact.get("context_tokens") or 0)
        prefix = int(fact.get("resident_prefix_layers")
                     or manager.model.layers)
        location = str(fact.get("location") or "local_hbm")
        if not session_id or home is None or context <= 0:
            continue
        manager.prepare_prefill(
            session_id=session_id, target_instance_index=int(home),
            history_tokens=0, trigger_request_id="cold-base-seed")
        manager.expand_prefill(
            session_id=session_id, instance_index=int(home),
            context_tokens=context,
            trigger_request_id="cold-base-seed")
        manager.mark_complete(session_id, 0)
        if location == "partial_hbm_remote" and 0 < prefix < (
                manager.model.layers):
            manager._evict_suffix(
                manager._sessions[session_id], phase="completion",
                reason="cold_base_seed", trigger_request_id="cold-base-seed",
                layer_start=prefix)


def _cold_scheduler_shell(runtime):
    """冷首边界 planner_state 推导用最小调度器壳（bounded proof 同款
    ``__new__`` 装配；真实 KVCacheManager/topology/joint 注册表全来自
    hydrate 的 runtime——原构造器对象，无第二套公式）。"""
    from face_scheduler import deterministic_xy_route  # noqa: F401
    from joint.hbm_port_flow_registry import HbmPortFlowRegistry
    from joint.joint_cost_model import (
        CausalHorizonEstimator,
        JointHardwareRates,
        LinkFlowRegistry,
        ServiceFactors,
    )
    from joint.joint_config import parse_joint_config
    from joint.predictor_cost_accounting import PredictorCostAccounting
    from online.sh30_online_scheduler import (
        Sh30OnlineScheduler,
        _OnlineInstanceState,
        _PoolPortRegistry,
        _TASK_LOAD_CACHE_CAPACITY,
    )

    scheduler = Sh30OnlineScheduler.__new__(Sh30OnlineScheduler)
    scheduler.topology = runtime.topology
    scheduler.hardware = runtime.hardware
    scheduler.model = runtime.model
    scheduler.kv_manager = runtime.kv_manager
    scheduler.p_chunk = runtime.p_chunk
    scheduler._joint_horizon = CausalHorizonEstimator(
        cold_start_default_forward_steps=1)
    scheduler._joint_flows = LinkFlowRegistry()
    scheduler._pool_ports = _PoolPortRegistry()
    scheduler._hbm_ports = HbmPortFlowRegistry()
    scheduler._hbm_ports.attach_active_decode_provider(lambda rank: 0)
    scheduler._joint_factors = ServiceFactors()
    scheduler._joint_rates = runtime.rates
    scheduler._instance_edge_ports = {
        instance.index: tuple(sorted({
            runtime.kv_manager.nearest_edge(rank)
            for rank in instance.ranks}))
        for instance in runtime.topology.instances}
    scheduler.joint_config = runtime.joint_config
    scheduler._joint_mode = runtime.joint_config.scheduler_mode
    scheduler._prefill_task_cache = {}
    scheduler._decode_task_load_cache = {}
    scheduler._task_load_cache_capacity = _TASK_LOAD_CACHE_CAPACITY
    scheduler._quota_tracker = None
    scheduler._link_telemetry_rates = {}
    scheduler._link_telemetry_flow_counts = {}
    scheduler._telemetry_last_window_bounds_ns = None
    scheduler.predictor_cost = PredictorCostAccounting()
    scheduler.instances = [
        _OnlineInstanceState(index=instance.index)
        for instance in runtime.topology.instances]
    # __new__ 壳对齐 __init__ 初值（F6 销账口径：直访属性漏设 =
    # AttributeError，不静默）。
    scheduler._snapshot_verify = False
    scheduler._kv_ledger_epoch = 0
    scheduler._admission_decision_wall_ns_total = 0
    scheduler._admission_decision_wall_ns_max = 0
    scheduler._admission_decision_count = 0
    scheduler._telemetry_last_tick_ns = 0
    return scheduler


def _derive_causal_metadata(fixture_state):
    """causal_metadata ← fixture_state["causal_input"]：只构造驱动作比对
    的 {context_tokens, horizon_tokens}（ctx 别名归一；不携带
    committed_decode_tokens——预测边界不得泄露未来执行状态）。"""
    causal = fixture_state.get("causal_input")
    if not isinstance(causal, dict):
        raise AcceptanceProjectionIncomplete(
            "fixture_state.causal_input is required to derive boundary "
            "causal_metadata")
    context = causal.get("context_tokens", causal.get("ctx"))
    horizon = causal.get("horizon_tokens")
    if isinstance(context, bool) or not isinstance(context, int) \
            or context < 1:
        raise AcceptanceProjectionIncomplete(
            "causal_input context tokens must be a positive integer")
    if isinstance(horizon, bool) or not isinstance(horizon, int) \
            or horizon < 1:
        raise AcceptanceProjectionIncomplete(
            "causal_input horizon_tokens must be a positive integer")
    return {"context_tokens": int(context), "horizon_tokens": int(horizon)}


def _resolve_instance_index(runtime, fixed_instance_id):
    """fixed_instance_id（字符串）→ 拓扑实例索引（名称或十进制索引）。"""
    text = str(fixed_instance_id)
    for instance in runtime.topology.instances:
        if instance.name == text:
            return int(instance.index)
    if text.lstrip("-").isdigit():
        index = int(text)
        for instance in runtime.topology.instances:
            if int(instance.index) == index:
                return index
    raise AcceptanceProjectionIncomplete(
        f"fixed_instance_id {fixed_instance_id!r} does not name any "
        "hydrated topology instance")


def _derive_cold_planner_state_P(
        runtime, shell, cost_model, *, request_id, causal_metadata,
        input_tokens=None):
    """冷首 P 边界 planner_state：session/request views 由 hydrate 的
    runtime + causal_input + 场景参数构造（bounded proof 的生产配方：
    session 视图走原 ``_joint_session_view``——基会话已按
    initial_state.history 的已提交事实用原构造器播种（裁决 2026-10-06
    路线 (i)）；RequestView 字段按 ``_try_admit_request`` 同款推导）。

    ``input_tokens`` = 本轮输入 token 数（已提交事实——事件带 arrival
    决策 payload 的 prefill_length）；缺省（无事件带/无事实）退化为
    ctx 全量输入的无基冷首形态（真实冷首请求，零适用/不适用如实）。"""
    from face_scheduler import kv_cache_shard_bytes_for_tokens
    from joint.joint_config import (
        REMOTE_READ_QUERY_TILE_TOKENS,
        prefill_remote_read_scan_passes,
    )
    seeded = _seeded_session_ids(shell)
    session_id = (
        seeded[0] if seeded else f"session:{request_id}")
    if input_tokens is None:
        input_tokens = int(causal_metadata["context_tokens"])
    input_tokens = int(input_tokens)
    session_view = shell._joint_session_view(session_id)
    horizon_tokens = int(causal_metadata["horizon_tokens"])
    input_kv_shards = kv_cache_shard_bytes_for_tokens(
        runtime.model, input_tokens, runtime.kv_manager.tp_degree)
    prefill_scan_passes = prefill_remote_read_scan_passes(
        input_tokens, runtime.p_chunk, REMOTE_READ_QUERY_TILE_TOKENS)
    session_fields = {
        "session_id": session_view.session_id,
        "home_instance": session_view.home_instance,
        "resident_instance": session_view.resident_instance,
        "location": session_view.location,
        "history_tokens": int(session_view.history_tokens),
        "resident_prefix_layers": int(session_view.resident_prefix_layers),
        "history_bytes_by_tp_rank": [
            int(value) for value in session_view.history_bytes_by_tp_rank],
        "missing_bytes_by_tp_rank": [
            int(value) for value in session_view.missing_bytes_by_tp_rank],
    }
    request_fields = {
        "request_id": str(request_id),
        "session_id": session_id,
        "input_tokens": input_tokens,
        "history_tokens_before": int(session_view.history_tokens),
        "estimated_decode_tokens": horizon_tokens,
        "horizon_source": "cold_start_default",
        "input_kv_bytes_by_tp_rank": [
            int(value) for value in input_kv_shards],
        "prefill_scan_passes": prefill_scan_passes,
        "prefill_chunk_tokens": int(runtime.p_chunk),
        "query_tile_tokens": int(REMOTE_READ_QUERY_TILE_TOKENS),
    }
    return {
        "version": PLANNER_STATE_VERSION,
        "stage_input": {
            "kind": "P",
            "session_view": session_fields,
            "request_view": request_fields,
            "prep_frontier_by_instance": {},
            "remote_enabled": bool(runtime.joint_config.remote_enabled),
        },
        "instance_loads": [
            export_instance_load_view(load)
            for _, load in sorted(cost_model.loads.items())],
        "registries": export_registries(shell, cost_model),
        "forecast_generator": export_forecast_generator(
            context_tokens=int(causal_metadata["context_tokens"]),
            horizon_tokens=horizon_tokens),
    }


def _derive_cold_planner_state_D(
        runtime, shell, cost_model, *, request_id, causal_metadata,
        preceding_boundary, pipeline_view, decision_epoch=1,
        prior_d_planner=None):
    """冷 D 边界 planner_state：由 tape 的 fixed_action/固定实例 + 已提交
    前缀（同一 result 中先行 P 边界 + 已重放 ledger）构造。

    完整型 P 前缀（stay/copy/recompute）：工作副本 = H_B 全量逐 rank 字节
    （模型派生），基础恒空。增量型（P=remote-read）需要 U/基础区间拆分，
    冷 fixture 状态不携带——按 §19.1 保持 state_mismatch/INCOMPLETE 语义，
    不伪造。

    多边界递进（主 agent 指令 2026-10-06）：后续 D 边界（同一 decode 相
    的后续列车提交）以首 D 边界的布局为已提交事实（边界事务恰一次），
    decision_epoch 随已提交边界数递增，继承义务取**当前**重放前缀的
    pipeline obligations（逐边界刷新的已提交事实）；decode_steps 保持
    因果估计口径（estimator 下限 = 预测 horizon——执行真值
    committed_decode_tokens 按 §19.1 不得泄入预测边界），工作副本字节
    保持因果可知的 H_B 口径（执行侧增长属执行真值，不进预测面）。
    ``prior_d_planner`` 在场时校验布局一致性后仅刷新以上三项。"""
    from face_scheduler import kv_cache_shard_bytes_for_tokens
    p_action = preceding_boundary.get("fixed_action")
    increment_layout = p_action == "remote-read"
    if p_action not in ("stay", "recompute", "copy", "remote-read"):
        raise AcceptanceProjectionIncomplete(
            f"cold D planner derivation requires a known P prefix action; "
            f"got {p_action!r} (state_mismatch semantics, nothing "
            "fabricated)")
    input_tokens = 0
    if increment_layout:
        # 增量型（多边界任务补全，2026-10-06）：U = 本轮输入 token 数，
        # 取自 P graph_commit 批的已提交事实——首 D 批 decode 读流
        # descriptor 的 tile 轴 total_units 是基础区间 token 数（H），
        # 工作副本 U = 边界 history_tokens − H。这里 H 直接采用
        # causal context（冷夹具的 base 区间基数）；U 需要真实输入量，
        # 由调用方（canonicalize 从 tape 的 arrival 决策事件 payload
        # prefill_length）供给——纯已提交事实，不虚构。
        input_tokens = int(pipeline_view.get("_cold_input_tokens", 0))
        if input_tokens <= 0:
            raise AcceptanceProjectionIncomplete(
                "cold D planner derivation for the increment layout "
                "(P=remote-read) requires the P turn's committed input "
                "token count; it is not present in the raw frames — "
                "the canonicalizer reads it from the event tape's "
                "arrival decision payload (prefill_length); state_"
                "mismatch semantics, nothing fabricated")
    p_instance = _resolve_instance_index(
        runtime, preceding_boundary["fixed_instance_id"])
    context = int(causal_metadata["context_tokens"])
    horizon = int(causal_metadata["horizon_tokens"])
    total_context = context + input_tokens
    working = kv_cache_shard_bytes_for_tokens(
        runtime.model,
        (input_tokens if increment_layout else context),
        runtime.kv_manager.tp_degree)
    base = (kv_cache_shard_bytes_for_tokens(
        runtime.model, context, runtime.kv_manager.tp_degree)
        if increment_layout else None)
    token_bytes = kv_cache_shard_bytes_for_tokens(
        runtime.model, 1, runtime.kv_manager.tp_degree)
    tp = len(working)
    seeded = _seeded_session_ids(shell)
    base_home = None
    if seeded:
        view = shell._joint_session_view(seeded[0])
        base_home = view.home_instance
    boundary_fields = {
        "session_id": (seeded[0] if seeded else f"session:{request_id}"),
        "home_instance": base_home,
        "p_instance_index": p_instance,
        "p_action": p_action,
        "history_tokens": total_context,
        "working_bytes_by_tp_rank": [int(value) for value in working],
        "base_bytes_by_tp_rank": (
            [0] * tp if base is None
            else [int(value) for value in base]),
        "decode_steps": horizon,
        "base_tokens": context,
        "kv_token_bytes": int(token_bytes[0]),
        "logic_home_instance": None,
        "kv_token_bytes_by_tp_rank": [
            int(value) for value in token_bytes],
        "working_intervals": [],
    }
    if prior_d_planner is not None:
        prior_view = prior_d_planner["stage_input"]["boundary_view"]
        for key in ("p_instance_index", "p_action", "history_tokens",
                    "base_tokens", "working_bytes_by_tp_rank",
                    "base_bytes_by_tp_rank"):
            if boundary_fields[key] != prior_view.get(key):
                raise AcceptanceProjectionIncomplete(
                    f"multi-boundary D derivation: committed layout fact "
                    f"{key!r} changed across decode-phase boundaries "
                    f"({boundary_fields[key]!r} != {prior_view.get(key)!r}); "
                    "the boundary transaction must commit exactly once — "
                    "refusing to re-derive a different layout (nothing "
                    "fabricated)")
    inherited = []
    obligations_view = (
        pipeline_view.get("obligations")
        if isinstance(pipeline_view, dict) else None)
    if isinstance(obligations_view, dict):
        inherited = sorted(str(key) for key in obligations_view)
    runtime_state_fields = {
        "terminated": False,
        "first_token": {
            "issued": False,
            "consumed": False,
            "rng_advances": 0,
            "rebuild_sampling_suppressed": False,
        },
        "obligations": [],
        "working_layers": [],
        "base_layers": [],
    }
    return {
        "version": PLANNER_STATE_VERSION,
        "stage_input": {
            "kind": "D",
            "boundary_view": boundary_fields,
            "runtime_state": runtime_state_fields,
            "decision_epoch": int(decision_epoch),
            "inherited_obligation_ids": inherited,
            "remote_enabled": bool(runtime.joint_config.remote_enabled),
        },
        "instance_loads": [
            export_instance_load_view(load)
            for _, load in sorted(cost_model.loads.items())],
        "registries": export_registries(shell, cost_model),
        "forecast_generator": export_forecast_generator(
            context_tokens=context,
            horizon_tokens=horizon),
    }


def _committed_input_tokens(fixture_state) -> int:
    """已提交事实：事件带 arrival 决策事件 payload.prefill_length（P 轮
    输入 token 数）。事件带与 state 同目录（驱动布局）；读取只取
    graph_commit 决策事件载荷，不构造任何预测面。"""
    base = getattr(fixture_state, "_verified_asset_root", None)
    if base is None:
        base = fixture_state.get("root")
    if base is None:
        return 0
    tape_path = Path(base) / "events.json"
    if not tape_path.is_file():
        return 0
    try:
        tape = json.loads(tape_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return 0
    for event in tape.get("events") or []:
        if event.get("kind") != "graph_commit":
            continue
        for decision in (event.get("payload") or {}).get(
                "decision_events") or []:
            payload = decision.get("payload") or {}
            if payload.get("prefill_length") is not None:
                return int(payload["prefill_length"])
    return 0


def _seeded_session_ids(shell) -> list:
    """基会话播种后 manager 在册的会话 id（因果可见事实）。"""
    manager = shell.kv_manager
    getter = getattr(manager, "session_ids", None)
    if getter is not None:
        try:
            return [str(item) for item in getter()]
        except TypeError:
            if isinstance(getter, property):
                return [str(item) for item in getter.fget(manager)]
    sessions = getattr(manager, "_sessions", None) or {}
    return [str(item) for item in sessions.keys()]


def _first_d_position(boundary_records) -> int:
    for index, record in enumerate(boundary_records):
        if record.get("stage") == "D":
            return index
    raise AcceptanceProjectionIncomplete(
        "multi-boundary D derivation lost its first D boundary")


def canonicalize_raw_frames(raw_result, fixture_state, *, source_root):
    """冷夹具 raw→canonical 规范化（执行文档 §19.1A Stage B）。

    输入 raw fixture result（boundary_records[].causal_prestate 与
    planner_state 均为 null、顶层 canonicalization_pending=true、
    audit_records 为**数组**且含 kind=="raw_v2_frame" 条目：C++
    publisher 的 V3 archive 或 V4 envelope 原文 + runtime_snapshot +
    producer_build）；fixture_state 为驱动 ``validate_fixture_state``
    验证态（精确键集）。输出同一 result dict（**去掉**
    canonicalization_pending——终检 result 为精确键集），填好每条边界
    的 causal_prestate 六域与 planner_state。

    1. 资产只经 fixture_state["assets"]（sha256 验字节）按原构造器
       hydrate；
    2. 同一协议版本的 V3 archive 或 V4 raw 帧与 boundary_records 按提交顺序 1:1 配对（同位序校验
       boundary_ordinal——与驱动 validate_raw_fixture_result 同规则），
       严格选择一个 V3 archive/V4 offline decoder并逐帧重放；
    3. work_ledger = 双账本导出的规范逻辑剩余视图合成；
    4. resources = 该帧 runtime_snapshot 原样（附 producer_build）；
    5. directory/capacity/history ← fixture_state["initial_state"] 的
       同名字段（loaded logical 冷初态）；causal_metadata ←
       causal_input 的 {context_tokens, horizon_tokens}；
    6. planner_state 由 hydrate 的 runtime + 冷调度器壳推导冷首边界
       （P：session/request views；D：先行 P 边界身份 + 已提交前缀），
       首边界之外的 P→D 后继按 §19.1 保持 state_mismatch/INCOMPLETE
       语义（明确异常，不伪造）。
    """
    import tempfile

    if not isinstance(raw_result, dict):
        raise AcceptanceProjectionIncomplete(
            "raw fixture result must be a JSON object")
    if not raw_result.get("canonicalization_pending"):
        raise AcceptanceProjectionIncomplete(
            "raw fixture result is not canonicalization_pending; refuse "
            "to re-canonicalize an already-canonical result")
    boundary_records = raw_result.get("boundary_records")
    if not isinstance(boundary_records, list) or not boundary_records:
        raise AcceptanceProjectionIncomplete(
            "boundary_records[] is required for Stage B canonicalization")
    for index, record in enumerate(boundary_records):
        if not isinstance(record, dict):
            raise AcceptanceProjectionIncomplete(
                f"boundary_records[{index}] must be a JSON object")
        if record.get("causal_prestate") is not None \
                or record.get("planner_state") is not None:
            raise AcceptanceProjectionIncomplete(
                "raw boundary_record.causal_prestate and planner_state "
                "must both be null in Stage A output (refusing "
                "pre-canonicalized input)")
    frames = _collect_raw_v2_frames(raw_result.get("audit_records"))
    pairs = _pair_frames_in_commit_order(frames, boundary_records)

    causal_metadata = _derive_causal_metadata(fixture_state)
    initial_state = fixture_state.get("initial_state")
    if not isinstance(initial_state, dict):
        raise AcceptanceProjectionIncomplete(
            "fixture_state.initial_state is required (cold loaded logical "
            "state validated by validate_cold_initial_state); refusing to "
            "synthesize directory/capacity/history")
    for field in ("directory", "capacity", "history"):
        if not isinstance(initial_state.get(field), dict):
            raise AcceptanceProjectionIncomplete(
                f"fixture_state.initial_state.{field} must be a JSON "
                "object (loaded logical cold initial state)")

    with tempfile.TemporaryDirectory(
            prefix="joint-stage-b-hydrate-") as private_dir:
        asset = _resolve_runtime_asset_strict(fixture_state)
        runtime = hydrate_runtime_asset(
            asset, private_dir=Path(private_dir))
        shell = _cold_scheduler_shell(runtime)
        _seed_cold_base_sessions(shell, initial_state)
        cost_model = shell._joint_cost_model(0)
        committed_input_tokens = _committed_input_tokens(fixture_state)

        # 增量重放（放量修复，2026-10-07）：单一 harness 逐帧应用、每
        # 边界一次快照——原"逐边界前缀重放"是 O(N^2)（53 帧案例实测
        # >300s：sum_i T(i)，T 随累计状态二次增长）；增量路径全序列
        # 实测 ~13s，快照每边界 ~6ms。语义等价：第 i 边界的视图仍 =
        # 前 i+1 帧按提交序经生产入口 `_apply_work_ledger` 的累积态。
        protocol_version = (pairs[0][0]["frame"]["protocol_version"]
                            if pairs else 3)
        replay_harness = _build_dual_ledger_harness(protocol_version)
        canonical_records = []
        for position, (frame_entry, record) in enumerate(pairs):
            frame = frame_entry["frame"]
            runtime_snapshot = frame_entry["runtime_snapshot"]
            replay_harness._apply_work_ledger({"work_ledger": frame})
            pipeline_view = _jsonify(
                replay_harness.joint_pipeline_ledger.remaining_view())
            work_ledger_view = {
                "certified_empty": False,
                "ordinary": replay_harness.joint_work_ledger.export_state(),
                "pipeline": pipeline_view,
                "replayed_frame_count": position + 1,
                "work_ledger_version": int(
                    replay_harness.joint_work_ledger.version),
                "work_ledger_watermark": int(
                    replay_harness.joint_work_ledger.event_watermark),
            }
            prestate = {
                "work_ledger": work_ledger_view,
                "resources": {
                    "backend_snapshot": runtime_snapshot,
                    "producer_build": frame_entry["producer_build"],
                },
                "directory": dict(initial_state["directory"]),
                "capacity": dict(initial_state["capacity"]),
                "obligations": {
                    "entries": (
                        sorted(str(key) for key in pipeline_view)
                        if isinstance(pipeline_view, dict) else []),
                    "pipeline": pipeline_view,
                },
                "history": dict(initial_state["history"]),
            }
            stage = record.get("stage")
            if stage == "P":
                if position != 0:
                    raise AcceptanceProjectionIncomplete(
                        "cold planner derivation covers the first boundary "
                        "only; a later P boundary is a P→D successor state "
                        "(state_mismatch semantics per exec doc 19.1; "
                        "nothing fabricated)")
                planner_state = _derive_cold_planner_state_P(
                    runtime, shell, cost_model,
                    request_id=raw_result.get("request_id"),
                    causal_metadata=causal_metadata,
                    input_tokens=committed_input_tokens)
            elif stage == "D":
                if position == 0:
                    raise AcceptanceProjectionIncomplete(
                        "cold D boundary requires a committed P prefix in "
                        "the same result (state_mismatch semantics per "
                        "exec doc 19.1; nothing fabricated)")
                prior_d = next(
                    (row for row in canonical_records
                     if row["stage"] == "D"), None)
                if prior_d is not None:
                    # 多边界递进：布局锚 = 已提交的首 D 边界行（其
                    # preceding P 前缀已在首推导校验）；刷新项 = 边界
                    # epoch（已提交边界数）与当前重放前缀的义务面。
                    first_d = _first_d_position(boundary_records)
                    layout_anchor = boundary_records[first_d - 1]
                    progression_view = dict(pipeline_view or {})
                    progression_view["_cold_input_tokens"] = (
                        committed_input_tokens)
                    planner_state = _derive_cold_planner_state_D(
                        runtime, shell, cost_model,
                        request_id=raw_result.get("request_id"),
                        causal_metadata=causal_metadata,
                        preceding_boundary=layout_anchor,
                        pipeline_view=progression_view,
                        decision_epoch=position,
                        prior_d_planner=prior_d["planner_state"])
                else:
                    first_d_pipeline_view = dict(pipeline_view or {})
                    first_d_pipeline_view["_cold_input_tokens"] = (
                        committed_input_tokens)
                    planner_state = _derive_cold_planner_state_D(
                        runtime, shell, cost_model,
                        request_id=raw_result.get("request_id"),
                        causal_metadata=causal_metadata,
                        preceding_boundary=boundary_records[position - 1],
                        pipeline_view=first_d_pipeline_view)
            else:
                raise AcceptanceProjectionIncomplete(
                    f"boundary stage {stage!r} must be P or D")
            canonical_records.append({
                "request_id": record["request_id"],
                "stage": stage,
                "boundary_ordinal": int(record["boundary_ordinal"]),
                "tick_ns": int(record["tick_ns"]),
                "fixed_instance_id": record.get("fixed_instance_id"),
                "fixed_action": record.get("fixed_action"),
                "causal_prestate": prestate,
                "causal_metadata": dict(causal_metadata),
                "planner_state": planner_state,
            })

    result = dict(raw_result)
    result["boundary_records"] = canonical_records
    # 终检 result 为精确键集：Stage A 的 canonicalization_pending 标记在
    # Stage B 输出中移除（不再附加任何额外顶层键）。
    result.pop("canonicalization_pending", None)
    return result


__all__ = [
    "DTO_VERSION",
    "PLANNER_STATE_VERSION",
    "RUNTIME_ASSET_VERSION",
    "NO_EXOGENOUS_GENERATOR_IDENTITY",
    "AcceptanceProjectionIncomplete",
    "AcceptanceProjectionFailure",
    "export_runtime_asset",
    "export_instance_load_view",
    "export_service_factors",
    "export_registries",
    "export_forecast_generator",
    "export_planner_state_P",
    "export_planner_state_D",
    "export_causal_prestate",
    "export_boundary",
    "hydrate_runtime_asset",
    "hydrate_instance_loads",
    "hydrate_registries",
    "hydrate_session_view",
    "hydrate_request_view",
    "hydrate_boundary_view",
    "build_acceptance_cost_model",
    "evaluate_boundary",
    "canonicalize_raw_frames",
]
