#!/usr/bin/env python3
"""Small, reproducible bridge-to-ledger integration case for protocol v4.

This file deliberately uses the checked-in production scheduler shell and
GraphBatchBuilder. The generated tape is one small M01-like remote-read case
(history 600, input 12, two decode tokens, one token per decode train); it is not a
representative full-trace performance input. The paired C++ runs are launched
by the caller so a process-group guard can own their lifetime and budget.

Example input generation (does not start the C++ fixture):

  python3 joint/test_sparse_pipeline_integration.py \
      --fixture-dir /tmp/joint-micro-v4 --generate-only \
      --native-library /path/to/libjoint_projection.so

The comparison mode consumes the raw results produced by frozen R_current
v2 and the current V4 sender on that exact input. V3 inputs are optional,
explicit offline archive operands; they are never treated as current live
frames. All v2 handling in this file is an explicit test-only reference
adapter; the online consumer remains strict V4.
"""
from __future__ import annotations

import argparse
import copy
from contextlib import contextmanager
import hashlib
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any


WORKLOAD_DIR = Path(__file__).resolve().parents[1]
ONLINE_DIR = WORKLOAD_DIR / "online"
REPO_ROOT = Path(__file__).resolve().parents[4]
for _path in (str(WORKLOAD_DIR), str(ONLINE_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

MICRO_CASE_ID = "M01-micro-p2-d3-seed000"
REQUEST_ID = "r1"
SESSION_ID = "s"
HOME_INSTANCE = 0
EXEC_INSTANCE = 1
CONTEXT_TOKENS = 600
INPUT_TOKENS = 12
DECODE_TOKENS = 2
TILE_TOKENS = 256  # Production auto/default tile resolution for this model.
TILE_BYTES = "auto"
TRAIN_MAX_ITER = 1
HORIZON_TOKENS = 2
FOUR_ACTIONS = frozenset({"stay", "recompute", "copy", "remote-read"})
T2_MAX_BOUNDARIES = 8
HISTORICAL_D20_CONTEXT_TOKENS = 10_856
HISTORICAL_D20_TOTAL_TILES = 344
HISTORICAL_D20_ITERATIONS = 8

RAW_FRAME_AUDIT_KIND = "raw_v2_frame"  # Frozen outer audit discriminator.
REQUEST_JOURNAL_NAME = "request_journal.jsonl"
WORK_LEDGER_SIDECAR_NAME = "request_journal_work_ledger.jsonl.gz"


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")


def _load_driver():
    driver_path = REPO_ROOT / "sh_test_mesh" / "run_scripts" / "joint_accel_acceptance.py"
    spec = importlib.util.spec_from_file_location(
        "joint_sparse_acceptance_driver", driver_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load fixture acceptance driver: {driver_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_native_library(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"same-ABI native projection library missing: {path}")
    import joint.native_projection as native_module
    from joint.native_projection import NativeProjectionLibrary
    library = NativeProjectionLibrary(str(path))
    original = native_module.load_production_projection_library
    native_module.load_production_projection_library = lambda: library
    return native_module, original


def fixed_action_after_full_enumeration(original, *, target: int,
                                        action: str, label: str,
                                        observations: list):
    """Test-only fixed-action hook: price first, then select an existing row.

    It leaves selector inputs and candidate rows untouched.  In particular,
    it does not provide future candidates or alter forecasts before the real
    selector evaluates its complete action face.
    """
    def wrapped(*args, **kwargs):
        record = original(*args, **kwargs)
        candidates = tuple(record.candidates)
        actions = {str(candidate.action) for candidate in candidates}
        if not FOUR_ACTIONS.issubset(actions):
            raise AssertionError(
                f"{label} selector did not enumerate all four actions: {sorted(actions)}")
        matches = [candidate for candidate in candidates
                   if candidate.instance_index == target
                   and candidate.action == action
                   and candidate.applicable
                   and candidate.cost_ns is not None]
        if not matches:
            raise AssertionError(
                f"{label} fixed candidate {action}@{target} was not applicable")
        record.chosen = matches[0]
        record.instance_rule_note = (
            f"test-only fixed tape choice after complete {label} enumeration")
        observations.append({
            "stage": label,
            "chosen": record.chosen.action,
            "instance_index": record.chosen.instance_index,
            "candidates": record.as_dict()["candidates"],
        })
        return record
    return wrapped


@contextmanager
def fixed_action_policy_hooks(*, p_target: int, p_action: str,
                              d_target: int, d_action: str,
                              observations: list | None = None):
    """Install test-only P/D hooks for a real scheduler or online service.

    Each production selector runs to completion first.  The hook then picks an
    already-applicable candidate from the complete result and captures the
    original candidate rows, so it cannot add a candidate or inject forecast
    input.  The caller owns the live BridgeServer/scheduler lifetime.
    """
    import online.sh30_online_scheduler as sh30

    rows = observations if observations is not None else []
    old_p = sh30.select_instance_and_action
    old_d = sh30.select_decode_target_and_action
    sh30.select_instance_and_action = fixed_action_after_full_enumeration(
        old_p, target=p_target, action=p_action, label="P",
        observations=rows)
    sh30.select_decode_target_and_action = fixed_action_after_full_enumeration(
        old_d, target=d_target, action=d_action, label="D",
        observations=rows)
    try:
        yield rows
    finally:
        sh30.select_instance_and_action = old_p
        sh30.select_decode_target_and_action = old_d


def _shape_only_selector(*, target: int, action: str, label: str,
                         observations: list, continuation: bool = False):
    """Return a fixed record for D20 shape construction, without pricing.

    D20 is a graph-size construction probe, not an action-selection or timing
    result.  Keeping this hook separate from _force_enumerated_choice makes
    that evidence boundary explicit: the small paired micro case prices its
    complete action face; the large construction-only case only needs a
    stable action so the production GraphBatchBuilder can emit the requested
    344-total-tile descriptor as eight groups of 43 tiles.
    """
    from joint.joint_cost_model import ActionCandidate
    from joint.joint_scheduler import SelectionRecord

    candidate = ActionCandidate(
        instance_index=target,
        action=action,
        applicable=True,
        inapplicable_reason=None,
        cost_ns=0,
        is_continuation=continuation,
    )

    def fixed(**_kwargs):
        record = SelectionRecord(
            mode="joint",
            chosen=candidate,
            candidates=(candidate,),
            instance_rule_note=(
                f"test-only {label} shape selection; no candidate pricing"),
            remote_enabled=True,
        )
        observations.append({
            "stage": label,
            "chosen": candidate.action,
            "instance_index": candidate.instance_index,
            "candidate_enumeration": "skipped-for-construction-only",
        })
        return record

    return fixed


def _start_delivery(scheduler, tick: int, sequence: int) -> None:
    scheduler._start_batch({
        "schema_version": 1,
        "delivery_sequence": sequence,
        "delivery_epoch": sequence,
        "tick": tick,
        "deferred_from_tick": tick,
        "reasons": ["ARRIVAL"],
        "arrivals": [],
        "completed_groups": [],
        "completed_nodes": [],
        "affected_ranks": [],
        "snapshot_handle": {"epoch": sequence, "tick": tick},
    })


def _ack_delivery(scheduler, sequence: int) -> None:
    from online.online_scheduler_base import SCHEMA_VERSION
    scheduler.last_applied_sequence = sequence
    scheduler.on_commit_ack({
        "schema_version": SCHEMA_VERSION,
        "delivery_sequence": sequence,
        "batch_id": sequence,
    })


def _in_flight_state(scheduler):
    for state in scheduler.instances:
        if state.in_flight_train is not None:
            return state
    return scheduler.instances[EXEC_INSTANCE]


def _cold_initial_state(rank_count: int, mesh_rows: int, mesh_cols: int,
                        port_count: int) -> dict:
    undirected = (mesh_rows * (mesh_cols - 1)
                  + mesh_cols * (mesh_rows - 1))
    components = []
    for link_id in range(2 * undirected):
        components.append({
            "identity": {"kind": "noc_link", "link_id": link_id,
                         "fluid_version": 0},
            "version": 0,
            "state": {"active_flows": [], "propagation_tails": []},
        })
    for rank in range(rank_count):
        components.extend((
            {"identity": {"kind": "hbm_rank", "rank": rank},
             "version": 0, "state": {"hbm_jobs": []}},
            {"identity": {"kind": "node_store", "rank": rank},
             "version": 0, "state": {
                 "nodes": [], "ready": [], "deferred": [],
                 "pending_events": [], "watches": [], "descriptors": [],
                 "cursors": [], "buffers": [], "obligations": [],
                 "hardware_inflight": 0,
             }},
        ))
    for port_index in range(port_count):
        components.append({
            "identity": {"kind": "remote_port", "port_index": port_index},
            "version": 0, "state": {"pool_jobs": []},
        })
    return {
        "tick_ns": 0,
        "next_flow_id": 0,
        "next_tag": 1,
        "ranks": [{"rank": rank, "next_store_id": 1}
                  for rank in range(rank_count)],
        "components": components,
        "directory": {
            "loaded": True,
            "entries": [{
                "session_id": SESSION_ID,
                "holder_instance": HOME_INSTANCE,
                "role": "base",
                "token_start": 0,
                "token_end": CONTEXT_TOKENS,
                "layer_start": 0,
                "layer_end": 12,
                "note": "resident base prefix (pool-backed suffix evicted)",
            }],
            "inflight_copies": [],
        },
        "capacity": {"reservations": []},
        "history": {"sessions": [{
            "session_id": SESSION_ID,
            "home_instance": HOME_INSTANCE,
            "location": "partial_hbm_remote",
            "context_tokens": CONTEXT_TOKENS,
            "resident_prefix_layers": 12,
        }]},
    }


def _write_assets(case_dir: Path, scheduler, hardware, topology) -> list[dict]:
    import joint.acceptance_projection as ap
    from face_scheduler import physical_edge_ranks
    from generate_trace import RemoteMemoryConfig

    assets_dir = case_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    scheduler.model_name = "joint-sparse-micro"
    scheduler.position_encoding = "none"
    scheduler.remote_operand_loads = False
    scheduler.kv_reserve_context_tokens = 0
    scheduler.trace_granularity = "request_aggregated"
    scheduler.hardware_capacity_profile = "default"
    edge_npus = tuple(int(rank) for rank in physical_edge_ranks(hardware))
    scheduler._joint_remote_memory_config = RemoteMemoryConfig(
        path=assets_dir / "remote_memory.json",
        edge_npus=edge_npus,
        mesh_rows=int(hardware.mesh_rows),
        mesh_cols=int(hardware.mesh_cols),
        remote_mem_latency_ns=0,
        remote_mem_bw_gbps=50.0,
        logical_pool="joint-sparse-micro-pool",
    )

    group_by_asset = {
        "comm_group": {
            "1": {"dimensions": [1, 2], "ranks": [0, 1]},
            "2": {"dimensions": [1, 2], "ranks": [2, 3]},
        },
        "network": {
            "text": "# sparse micro fixture topology\n"
                    "topology: [ Line, Line ]\n"
                    "npus_count: [ 2, 2 ]\n"
                    "bandwidth: [ 2.0, 2.0 ]\n"
                    "latency: [ 0, 0 ]\n",
        },
        "remote_memory": {
            "memory-type": "PER_NPU_MEMORY_EXPANSION",
            "npu-ids": list(range(int(hardware.mesh_rows * hardware.mesh_cols))),
            "remote-mem-bw": 50.0,
            "remote-mem-latency": 0,
        },
        "system": {
            "scheduling-policy": "FIFO",
            "endpoint-delay": 1,
            "active-chunks-per-dimension": 1,
            "preferred-dataset-splits": 6,
            "all-reduce-implementation": ["ring", "ring"],
            "all-gather-implementation": ["ring", "ring"],
            "reduce-scatter-implementation": ["ring", "ring"],
            "all-to-all-implementation": ["ring", "ring"],
            "collective-optimization": "localBWAware",
            "roofline-enabled": 1,
            "replay-only": 0,
            "track-local-mem": 0,
            "trace-enabled": 0,
            "hbm-bandwidth-contention": 1,
            "peak-perf": int(hardware.peak_perf_tflops),
            "local-mem-bw": float(hardware.local_hbm_bandwidth_gbps),
            "local-mem-latency": int(hardware.local_hbm_latency_ns),
            "remote-mem-bw": 50.0,
            "remote-mem-latency": 0,
        },
    }
    for name, content in group_by_asset.items():
        path = assets_dir / ("network.yml" if name == "network"
                             else f"{name}.json")
        if name == "network":
            path.write_text(content["text"], encoding="utf-8")
        else:
            _write_json(path, content)
    _write_json(assets_dir / "python_runtime.json",
                ap.export_runtime_asset(scheduler))

    entries = []
    for asset_id, path in (
            ("comm_group", assets_dir / "comm_group.json"),
            ("network", assets_dir / "network.yml"),
            ("remote_memory", assets_dir / "remote_memory.json"),
            ("system", assets_dir / "system.json"),
            ("python_runtime", assets_dir / "python_runtime.json")):
        entries.append({
            "asset_id": asset_id,
            "path": path.relative_to(case_dir).as_posix(),
            "sha256": _sha256_file(path),
        })
    return entries


def _configure_scheduler_shape(helper, lifecycle, *, context_tokens: int,
                                input_tokens: int, decode_tokens: int,
                                train_max_iter: int) -> dict:
    names = ("HISTORY_TOKENS", "INPUT_TOKENS", "DECODE_TOKENS", "TILE_BYTES")
    old = {name: getattr(lifecycle, name) for name in names}
    lifecycle.HISTORY_TOKENS = context_tokens
    lifecycle.INPUT_TOKENS = input_tokens
    lifecycle.DECODE_TOKENS = decode_tokens
    lifecycle.TILE_BYTES = str(TILE_BYTES)
    helper.TILE_BYTES = str(TILE_BYTES)
    return old


def _build_production_batches(*, native_library: Path | None,
                              context_tokens: int = CONTEXT_TOKENS,
                              input_tokens: int = INPUT_TOKENS,
                              decode_tokens: int = DECODE_TOKENS,
                              train_max_iter: int = TRAIN_MAX_ITER,
                              construction_only: bool = False):
    """Build actual P/D batches from the current in-repo production path."""
    helper = importlib.import_module(
        "online.test_pipeline_descriptor_submission")
    lifecycle = importlib.import_module(
        "online.test_prefill_remote_read_lifecycle")
    old_helper_tile_bytes = helper.TILE_BYTES
    old_globals = _configure_scheduler_shape(
        helper, lifecycle, context_tokens=context_tokens,
        input_tokens=input_tokens, decode_tokens=decode_tokens,
        train_max_iter=train_max_iter)
    native_module = None
    original_loader = None
    builder = None
    observations: list[dict] = []
    try:
        if not construction_only:
            if native_library is None:
                raise ValueError(
                    "full candidate enumeration requires --native-library")
            native_module, original_loader = _load_native_library(native_library)
        builder = helper.PublishedBatchProductionPathTest(methodName="runTest")
        original_manager_factory = helper._make_manager
        if construction_only:
            def _large_shape_manager(model):
                from dataclasses import replace
                from face_scheduler import (
                    FaceInstanceSpec, KVCacheManager, build_instances)

                _, prototype_topology, _ = original_manager_factory(model)
                hardware = replace(
                    prototype_topology.hardware,
                    local_hbm_capacity_bytes=64 * (1 << 30))
                specs = tuple(FaceInstanceSpec(
                    instance.name, instance.pg_name, tuple(instance.ranks))
                    for instance in prototype_topology.instances)
                topology = build_instances(hardware, specs)
                manager = KVCacheManager(
                    topology, model, category_mode="typed",
                    layer_policy="adaptive", pool_bandwidth_gbps=4.0,
                    pool_latency_ns=10)
                return hardware, topology, manager

            helper._make_manager = _large_shape_manager
        try:
            builder.setUp()
        finally:
            helper._make_manager = original_manager_factory
        scheduler = builder.scheduler
        runtime = builder.runtime

        from joint.joint_config import parse_joint_config
        scheduler.joint_config = parse_joint_config({
            "JOINT_REMOTE_READ_TILE_BYTES": TILE_BYTES,
            "JOINT_ALLOWED_ACTIONS": "all",
        })
        scheduler._joint_mode = scheduler.joint_config.scheduler_mode
        scheduler._train_max_iter = train_max_iter
        # The production constructor normally injects resolved protocol values
        # into KVCacheManager. This test chassis uses __new__, so wire the same
        # auto/default values before admission and graph creation.
        from face_scheduler import kv_cache_shard_bytes_for_tokens
        from joint.joint_config import (
            REMOTE_READ_TILE_AUTO,
            resolve_remote_read_tile_tokens,
        )
        remote_read_kv_token_bytes = kv_cache_shard_bytes_for_tokens(
            scheduler.model, 1, scheduler.topology.instances[0].size)[0]
        scheduler.kv_manager._remote_read_kv_token_bytes = (
            remote_read_kv_token_bytes)
        scheduler.kv_manager._remote_read_tile_tokens = (
            resolve_remote_read_tile_tokens(
                REMOTE_READ_TILE_AUTO, remote_read_kv_token_bytes))
        if int(scheduler.kv_manager._remote_read_tile_tokens) != TILE_TOKENS:
            raise AssertionError(
                "this fixture expects the production auto/default 256-token "
                "tile, got "
                f"{scheduler.kv_manager._remote_read_tile_tokens}")
        # Match a fully initialized scheduler's bounded delivery/ACK ledgers.
        scheduler._acked_through = -1
        scheduler._acked_out_of_order = set()
        scheduler.last_applied_sequence = -1
        scheduler.ack_count = 0
        scheduler.delivery_count = 0
        scheduler.committed_kv_action_batches = 0
        scheduler.ledger_committed = {}
        scheduler._delivery_reply_cache = None

        import online.sh30_online_scheduler as sh30
        original_p = sh30.select_instance_and_action
        if construction_only:
            sh30.select_instance_and_action = _shape_only_selector(
                target=EXEC_INSTANCE, action="remote-read", label="P",
                observations=observations)
        else:
            sh30.select_instance_and_action = fixed_action_after_full_enumeration(
                original_p, target=EXEC_INSTANCE, action="remote-read",
                label="P", observations=observations)
        try:
            admitted = scheduler._try_admit_request(runtime, 500)
        finally:
            sh30.select_instance_and_action = original_p
        if not admitted:
            raise RuntimeError("production P admission did not accept micro case")
        scheduler.runtime_by_request_id[REQUEST_ID] = runtime
        scheduler.runtimes.append(runtime)
        scheduler._runtime_index[REQUEST_ID] = 0

        _start_delivery(scheduler, 600, 0)
        scheduler._plan_and_emit_trains(600)
        from online.online_scheduler_base import OnlineSchedulerBase
        p_batch = OnlineSchedulerBase.build_graph_batch(scheduler)
        if not p_batch["nodes"] or not p_batch["pipeline_descriptors"]:
            raise AssertionError("production P builder emitted no nodes/descriptors")

        p_state = scheduler.instances[EXEC_INSTANCE]
        p_plan = p_state.in_flight_train
        if p_plan is None:
            raise AssertionError("production P builder has no in-flight train")
        if int(scheduler.p_chunk) != 512:
            raise AssertionError(
                f"fixture chassis expected p_chunk=512, got {scheduler.p_chunk}")
        if (int(runtime.history_tokens_before) != context_tokens
                or int(runtime.joint_input_tokens) != input_tokens
                or int(runtime.prefill_tokens_to_process) != input_tokens):
            raise AssertionError(
                "runtime P geometry differs from the requested history/input "
                f"shape: H={runtime.history_tokens_before}, "
                f"U={runtime.joint_input_tokens}, "
                f"work={runtime.prefill_tokens_to_process}")
        if REQUEST_ID not in p_plan["drain_members"]:
            raise AssertionError(
                "P plan did not drain the one-chunk fixture request")
        scheduler._finalize_completed_trains(
            tuple(p_plan["drain_members"]), (),
            tuple([p_plan["train_id"]] if p_plan["sentinel"] else []), 900)
        if (int(runtime.remaining_chunks) != 0
                or int(runtime.prefill_tokens_completed)
                != int(runtime.prefill_tokens_to_process)):
            raise AssertionError(
                "P construction did not complete the fixture's sole prefill "
                f"chunk: remaining={runtime.remaining_chunks}, "
                f"completed={runtime.prefill_tokens_completed}/"
                f"{runtime.prefill_tokens_to_process}")
        runtime.queued_chunk_load_ns = 0

        original_d = sh30.select_decode_target_and_action
        if construction_only:
            sh30.select_decode_target_and_action = _shape_only_selector(
                target=EXEC_INSTANCE, action="remote-read", label="D",
                observations=observations, continuation=True)
        else:
            sh30.select_decode_target_and_action = fixed_action_after_full_enumeration(
                original_d, target=EXEC_INSTANCE, action="remote-read",
                label="D", observations=observations)
        try:
            scheduler._on_prefill_drain(REQUEST_ID, 1000)
        finally:
            sh30.select_decode_target_and_action = original_d

        decode_batches = []
        decode_train_iterations = []
        tick = 1100
        sequence = 1
        _start_delivery(scheduler, tick, sequence)
        scheduler._plan_and_emit_trains(tick)
        first_decode_plan = _in_flight_state(scheduler).in_flight_train
        if first_decode_plan is not None:
            decode_train_iterations.append(int(first_decode_plan["iterations"]))
        decode_batches.append(OnlineSchedulerBase.build_graph_batch(scheduler))
        _ack_delivery(scheduler, sequence)

        cap = max(4, decode_tokens // max(1, train_max_iter) + 4)
        while (not runtime.completed
               and runtime.decode_tokens_consumed < decode_tokens
               and len(decode_batches) < cap):
            train_state = _in_flight_state(scheduler)
            plan = train_state.in_flight_train
            if plan is None:
                break
            remaining = decode_tokens - runtime.decode_tokens_consumed
            completed_now = ((REQUEST_ID,)
                             if remaining <= train_max_iter else ())
            scheduler._finalize_completed_trains(
                tuple(plan["drain_members"]), completed_now,
                tuple([plan["train_id"]] if plan["sentinel"] else []),
                tick + 100)
            if completed_now:
                scheduler._complete_requests([REQUEST_ID], tick + 200)
                tick += 300
                sequence += 1
                _start_delivery(scheduler, tick, sequence)
                decode_batches.append(
                    OnlineSchedulerBase.build_graph_batch(scheduler))
                _ack_delivery(scheduler, sequence)
                break
            tick += 300
            sequence += 1
            _start_delivery(scheduler, tick, sequence)
            scheduler._plan_and_emit_trains(tick)
            next_decode_plan = _in_flight_state(scheduler).in_flight_train
            if next_decode_plan is not None:
                decode_train_iterations.append(
                    int(next_decode_plan["iterations"]))
            decode_batches.append(
                OnlineSchedulerBase.build_graph_batch(scheduler))
            _ack_delivery(scheduler, sequence)

        if len(decode_batches) < 2:
            raise AssertionError(
                "micro builder must produce at least two real D batches")
        if not any(batch["nodes"] for batch in decode_batches):
            raise AssertionError("production D builder emitted no nodes")
        return {
            "p_batch": p_batch,
            "decode_batches": decode_batches,
            "scheduler": scheduler,
            "runtime": runtime,
            "hardware": scheduler.hardware,
            "topology": scheduler.topology,
            "observations": observations,
            "decode_train_iterations": decode_train_iterations,
        }
    finally:
        for name, value in old_globals.items():
            setattr(lifecycle, name, value)
        helper.TILE_BYTES = old_helper_tile_bytes
        if native_module is not None:
            native_module.load_production_projection_library = original_loader
        if builder is not None:
            # The helper is a TestCase chassis used only for its production
            # constructors; release cleanups should remain explicit.
            builder.doCleanups()


def _batch_for_tape(batch: dict) -> dict:
    result = dict(batch)
    result.pop("kernel_contract", None)
    return result


def _descriptor_shape_rows(batches: list[dict]) -> list[dict]:
    """Report compact descriptor groups and affine run domains."""
    rows = []
    for batch_index, batch in enumerate(batches):
        for descriptor_index, descriptor in enumerate(
                batch.get("pipeline_descriptors", [])):
            identity = descriptor.get("identity", {})
            groups = []
            for group in descriptor.get("groups", []):
                axis = group["tile_axis"]
                groups.append({
                    "group_ordinal": int(group["group_ordinal"]),
                    "pass_or_window_ordinal": int(
                        group["pass_or_window_ordinal"]),
                    "source_ordinal": int(group["source_ordinal"]),
                    "layer_start": int(group["layer_start"]),
                    "layer_end": int(group["layer_end"]),
                    "tile_axis": {
                        key: axis[key] for key in (
                            "unit", "total_units", "units_per_tile",
                            "tile_count", "tail_units",
                            "global_tile_first")},
                })
            runs = descriptor.get("endpoint_runs", [])
            groups_by_id = {int(group["group_ordinal"]): group
                            for group in descriptor.get("groups", [])}
            def _run_row(run):
                return {
                    key: run[key] for key in (
                        "rank", "group_ordinal", "first_tile", "tile_count")
                } | {
                    "first_json_id": run.get("first_json_id"),
                    "tile_stride": run.get("tile_stride"),
                    "endpoints": [leg["endpoint"]
                                  for leg in run.get("legs", [])],
                }
            def _global_first(run):
                group = groups_by_id[int(run["group_ordinal"])]
                return (int(group["tile_axis"]["global_tile_first"])
                        + int(run["first_tile"]) - 1)
            rows.append({
                "batch_index": batch_index,
                "descriptor_index": descriptor_index,
                "stage": identity.get("stage"),
                "rank": identity.get("rank"),
                "segment_id": identity.get("segment_id"),
                "buffer_domain_id": identity.get("buffer_domain_id"),
                "stream_kind": descriptor.get("stream_kind"),
                "protocol": {
                    key: descriptor.get("protocol", {}).get(key)
                    for key in ("buffer_slots", "release_lag_tiles",
                                "initial_tiles_without_release")},
                "group_count": len(groups),
                "groups": groups,
                "tile_count_sum_across_groups": sum(
                    row["tile_axis"]["tile_count"] for row in groups),
                "endpoint_run_count": len(runs),
                "endpoint_run_tile_count_sum": sum(
                    int(run["tile_count"]) for run in runs),
                "endpoint_runs": [
                    {**_run_row(run),
                     "global_first_tile": _global_first(run)}
                    for run in runs],
                "endpoint_dependency_run_count": len(
                    descriptor.get("endpoint_dependency_runs", [])),
                "consume_completion_run_count": len(
                    descriptor.get("consume_completion_runs", [])),
                "logical_id_range_count": len(
                    descriptor.get("logical_id_ranges", [])),
                "anchor_count": len(descriptor.get("anchors", [])),
            })
    return rows


def _assert_group_tile_count(shape_rows: list[dict], *, stream_kind: str,
                             expected: int, label: str) -> None:
    axes = [group["tile_axis"]
            for descriptor in shape_rows
            if descriptor["stream_kind"] == stream_kind
            for group in descriptor["groups"]]
    if not axes:
        raise AssertionError(f"{label} has no {stream_kind} group tile axes")
    counts = [int(axis["tile_count"]) for axis in axes]
    if any(count != expected for count in counts):
        raise AssertionError(
            f"{label} expected {stream_kind} tile_count={expected}, got {counts}")


def _assert_decode_tile_three_reuse(batches: list[dict], *, label: str) -> dict:
    """Require stable per-rank domains and each group's third-tile lag-2 release."""
    descriptors = []
    per_batch_domains = []
    for batch_index, batch in enumerate(batches):
        rows = [descriptor
                for descriptor in batch.get("pipeline_descriptors", [])
                if descriptor.get("stream_kind") == "decode_remote_credit"]
        if not rows:
            raise AssertionError(
                f"{label} D batch {batch_index} has no decode credit descriptor")
        domains_by_rank = {}
        for descriptor in rows:
            identity = descriptor.get("identity", {})
            rank = int(identity["rank"])
            domain = str(identity["buffer_domain_id"])
            previous = domains_by_rank.setdefault(rank, domain)
            if previous != domain:
                raise AssertionError(
                    f"{label} D batch {batch_index} has multiple buffer "
                    f"domains for rank {rank}: {previous!r}, {domain!r}")
        per_batch_domains.append(domains_by_rank)
        descriptors.extend(rows)
    if not descriptors:
        raise AssertionError(f"{label} has no decode credit descriptors")
    domains_by_rank = per_batch_domains[0]
    for batch_index, observed in enumerate(per_batch_domains[1:], start=1):
        if observed != domains_by_rank:
            raise AssertionError(
                f"{label} changed per-rank buffer domains between D batches: "
                f"first={domains_by_rank}, batch{batch_index}={observed}")
    release_tile_three = False
    release_dependency_tile_three = False
    for descriptor in descriptors:
        groups = {int(group["group_ordinal"]): group
                  for group in descriptor.get("groups", [])}
        protocol = descriptor.get("protocol", {})
        if (int(protocol.get("buffer_slots", 0)) != 2
                or int(protocol.get("release_lag_tiles", 0)) != 2):
            raise AssertionError(
                f"{label} descriptor does not retain the two-slot lag-2 protocol")
        for run in descriptor.get("endpoint_runs", []):
            group = groups[int(run["group_ordinal"])]
            group_tile_count = int(group["tile_axis"]["tile_count"])
            local_first = int(run["first_tile"])
            local_last = local_first + int(run["tile_count"]) - 1
            has_release_ack = any(
                leg.get("endpoint") in ("release_ack_send", "release_ack_recv")
                for leg in run.get("legs", []))
            if (group_tile_count >= 3 and has_release_ack
                    and local_first <= 3 <= local_last):
                release_tile_three = True
        for run in descriptor.get("endpoint_dependency_runs", []):
            group = groups[int(run["group_ordinal"])]
            group_tile_count = int(group["tile_axis"]["tile_count"])
            local_first = int(run["first_tile"])
            local_last = local_first + int(run["tile_count"]) - 1
            if (run.get("endpoint") in
                    ("release_ack_send", "release_ack_recv")
                    and group_tile_count >= 3
                    and local_first <= 3 <= local_last):
                release_dependency_tile_three = True
    if not release_tile_three or not release_dependency_tile_three:
        raise AssertionError(
            f"{label} lacks the tile-3 lag-2 release ACK/dependency; "
            f"endpoint={release_tile_three}, dependency="
            f"{release_dependency_tile_three}")
    return {
        "buffer_domains_by_rank": {
            str(rank): domain for rank, domain in sorted(domains_by_rank.items())},
        "per_group_tile_three_release_ack": True,
        "per_group_tile_three_release_dependency": True,
        "buffer_slots": 2,
        "release_lag_tiles": 2,
    }


def _write_micro_case(case_dir: Path, native_library: Path) -> dict:
    case_dir.mkdir(parents=True, exist_ok=True)
    drive = _build_production_batches(native_library=native_library)
    p_batch = drive["p_batch"]
    decode_batches = drive["decode_batches"]
    scheduler = drive["scheduler"]
    hardware = drive["hardware"]
    topology = drive["topology"]
    if INPUT_TOKENS > int(scheduler.p_chunk):
        raise AssertionError(
            "micro input must fit in one production P chunk: "
            f"{INPUT_TOKENS} > {scheduler.p_chunk}")
    contract = dict(p_batch["kernel_contract"])
    for batch in decode_batches:
        if batch["kernel_contract"] != contract:
            raise AssertionError("P/D kernel contract drifted inside micro case")
    shape_rows = _descriptor_shape_rows([p_batch, *decode_batches])
    _assert_group_tile_count(
        shape_rows, stream_kind="prefill_read_window", expected=2,
        label="600-token-history auto-tile micro P graph")
    _assert_group_tile_count(
        shape_rows, stream_kind="decode_remote_credit", expected=3,
        label="600-token-history auto-tile micro D graph")
    decode_axes = [group["tile_axis"]
                   for descriptor in shape_rows
                   if descriptor["stream_kind"] == "decode_remote_credit"
                   for group in descriptor["groups"]]
    if any(axis["unit"] != "source_tokens"
           or int(axis["total_units"]) != CONTEXT_TOKENS
           or int(axis["units_per_tile"]) != TILE_TOKENS
           or int(axis["tail_units"]) != CONTEXT_TOKENS - 2 * TILE_TOKENS
           for axis in decode_axes):
        raise AssertionError(
            "micro D axes must expose the production 600-token base source "
            "as 256, 256, 88 token tiles")
    d_batches = [batch for batch in decode_batches
                 if any(descriptor.get("stream_kind") ==
                        "decode_remote_credit"
                        for descriptor in batch.get("pipeline_descriptors", []))]
    slot_reuse = _assert_decode_tile_three_reuse(
        d_batches, label="600-token-history auto-tile micro D graph")

    entries = _write_assets(case_dir, scheduler, hardware, topology)
    rank_count = int(hardware.mesh_rows * hardware.mesh_cols)
    edge_npus = tuple(int(x) for x in scheduler._joint_remote_memory_config.edge_npus)
    asset_ids = {entry["asset_id"] for entry in entries}
    if asset_ids != {"comm_group", "network", "remote_memory", "system",
                     "python_runtime"}:
        raise AssertionError(f"unexpected fixture assets: {sorted(asset_ids)}")

    state = {
        "version": "execution-fixture-v1-20261006",
        "kind": "cold_initial",
        "case_id": MICRO_CASE_ID,
        "cluster_id": "M01/micro-p2-d3-seed000",
        "scenario_id": "M01",
        "seed": 0,
        "request_id": REQUEST_ID,
        "fixed_instance_id": str(EXEC_INSTANCE),
        "fixed_action": "remote-read",
        "causal_input": {
            "focus_stage": "D",
            "context_tokens": CONTEXT_TOKENS,
            "horizon_tokens": HORIZON_TOKENS,
            "competition": "low",
            "source_count": 1,
            "committed_decode_tokens": DECODE_TOKENS,
        },
        "assets": entries,
        "hardware": {
            "rank_count": rank_count,
            "system_asset_id": "system",
            "network_asset_id": "network",
            "remote_memory_asset_id": "remote_memory",
            "die_hardware_asset_id": None,
        },
        "initial_state": _cold_initial_state(
            rank_count, int(hardware.mesh_rows), int(hardware.mesh_cols),
            len(edge_npus)),
        "python_runtime_asset_id": "python_runtime",
    }

    events = []
    ordinal = 0
    p_descriptors = p_batch["pipeline_descriptors"]
    if (not p_descriptors
            or any(row.get("stream_kind") != "prefill_read_window"
                   for row in p_descriptors)):
        raise AssertionError(
            "micro P batch must contain only prefill-read descriptors")
    # pipeline_execution_fixture_main identifies each committed stage by the
    # final descriptor in the batch, so its terminal gate must use that same
    # canonical per-rank segment ID.
    p_segment = str(p_descriptors[-1]["identity"]["segment_id"])

    def add(trigger, kind, payload):
        nonlocal ordinal
        events.append({"ordinal": ordinal, "trigger": trigger,
                       "kind": kind, "payload": payload})
        ordinal += 1

    p_arrival = {
        "seq": 1,
        "reason": "ARRIVAL",
        "request_id": REQUEST_ID,
        "stage": "prefill",
        "generation": 0,
        "payload": {
            "session_id": SESSION_ID,
            "turn_index": 1,
            "prefill_length": INPUT_TOKENS,
            "decode_length": DECODE_TOKENS,
            "ingress_seq": 1,
            "queue_index": 0,
        },
    }
    p_boundary = {
        "stage": "P",
        "fixed_instance_id": str(EXEC_INSTANCE),
        "fixed_action": "remote-read",
        "causal_metadata": {
            "context_tokens": CONTEXT_TOKENS,
            "horizon_tokens": HORIZON_TOKENS,
        },
    }
    add({"kind": "tick", "tick_ns": 0}, "graph_commit", {
        "batch": _batch_for_tape(p_batch),
        "decision_events": [p_arrival],
        "boundary_record": p_boundary,
    })

    for index, batch in enumerate(decode_batches):
        trigger = ({"kind": "gate_terminal", "segment_id": p_segment,
                    "role": "settle"}
                   if index == 0 else
                   {"kind": "tick", "tick_ns": 1100 + 300 * index})
        drain_event = {
            "seq": 2 + index,
            "reason": "PREFILL_DRAIN",
            "request_id": REQUEST_ID,
            "stage": "prefill",
            "generation": 0,
            "payload": {"watch_member_count": 1},
        }
        d_boundary = {
            "stage": "D",
            "fixed_instance_id": str(EXEC_INSTANCE),
            "fixed_action": "remote-read",
            "causal_metadata": {
                "context_tokens": CONTEXT_TOKENS,
                "horizon_tokens": HORIZON_TOKENS,
            },
        }
        add(trigger, "graph_commit", {
            "batch": _batch_for_tape(batch),
            "decision_events": [drain_event],
            "boundary_record": d_boundary,
        })
    add({"kind": "tick",
         "tick_ns": 1100 + 300 * len(decode_batches) + 5_000_000},
        "request_complete", {"request_id": REQUEST_ID})

    state_path = case_dir / "state.json"
    events_path = case_dir / "events.json"
    precision_path = case_dir / "precision.json"
    _write_json(state_path, state)
    _write_json(events_path, {
        "version": "execution-event-tape-v1-20261006",
        "case_id": MICRO_CASE_ID,
        "events": events,
    })
    _write_json(precision_path, contract)

    driver = _load_driver()
    driver.validate_fixture_state(json.loads(state_path.read_text()), state_path)
    driver.validate_event_tape(json.loads(events_path.read_text()),
                               case_id=MICRO_CASE_ID)

    source_files = {
        "builder_test": Path(importlib.import_module(
            "online.test_pipeline_descriptor_submission").__file__),
        "builder_lifecycle": Path(importlib.import_module(
            "online.test_prefill_remote_read_lifecycle").__file__),
        "graph_batch_builder": ONLINE_DIR / "graph_batch_builder.py",
        "online_scheduler_base": ONLINE_DIR / "online_scheduler_base.py",
        "sh30_online_scheduler": ONLINE_DIR / "sh30_online_scheduler.py",
        "acceptance_projection": Path(importlib.import_module(
            "joint.acceptance_projection").__file__),
        "pipeline_descriptor": WORKLOAD_DIR / "joint" / "pipeline_descriptor.py",
        "pipeline_ledger": WORKLOAD_DIR / "joint" / "pipeline_ledger.py",
    }
    source_hashes = {
        name: {"path": str(path.resolve()), "sha256": _sha256_file(path)}
        for name, path in source_files.items()
    }
    output_hashes = {
        str(path.relative_to(case_dir)): _sha256_file(path)
        for path in (state_path, events_path, precision_path,
                     *(case_dir / entry["path"] for entry in entries))
    }
    provenance = {
        "generator": str(Path(__file__).resolve()),
        "generator_sha256": _sha256_file(Path(__file__).resolve()),
        "source_hashes": source_hashes,
        "input_shape": {
            "scenario": "M01-like remote-read micro case",
            "context_tokens": CONTEXT_TOKENS,
            "input_tokens": INPUT_TOKENS,
            "prefill_context_tokens": CONTEXT_TOKENS + INPUT_TOKENS,
            "prefill_work_tokens": INPUT_TOKENS,
            "prefill_chunk_size": int(scheduler.p_chunk),
            "prefill_chunk_count": 1,
            "decode_tokens": DECODE_TOKENS,
            "tile_mode": TILE_BYTES,
            "tile_tokens": int(scheduler.kv_manager._remote_read_tile_tokens),
            "tile_token_bytes_per_rank": int(
                scheduler.kv_manager._remote_read_kv_token_bytes),
            "train_max_iter": TRAIN_MAX_ITER,
            "allowed_actions": "all",
            "fixed_action": "remote-read after all candidates were evaluated",
            "cpp_physical_event_loop": False,
        },
        "production_batches": {
            "prefill_nodes": len(p_batch["nodes"]),
            "prefill_descriptors": len(p_batch["pipeline_descriptors"]),
            "decode_batches": len(decode_batches),
            "decode_nodes": [len(batch["nodes"]) for batch in decode_batches],
            "decode_descriptors": [len(batch["pipeline_descriptors"])
                                   for batch in decode_batches],
            "descriptor_shapes": shape_rows,
            "decode_slot_reuse": slot_reuse,
            "complete_candidate_observations": drive["observations"],
        },
        "generated_sha256": output_hashes,
    }
    _write_json(case_dir / "provenance.json", provenance)
    return provenance


def _raw_frames(raw_result: dict, *, expected_schema: int) -> list[dict]:
    audit = raw_result.get("audit_records")
    if not isinstance(audit, list):
        raise ValueError("raw fixture audit_records must be an array")
    frames = []
    for entry in audit:
        if not isinstance(entry, dict):
            raise ValueError("raw fixture audit entry must be an object")
        if entry.get("kind") != RAW_FRAME_AUDIT_KIND:
            continue
        if set(entry) != {"kind", "boundary_ordinal", "frame",
                          "runtime_snapshot", "producer_build"}:
            raise ValueError("raw frame audit entry has an unexpected key set")
        frame = entry.get("frame")
        if not isinstance(frame, dict):
            raise ValueError("raw frame envelope must be an object")
        if (type(frame.get("schema_version")) is not int
                or type(frame.get("protocol_version")) is not int
                or frame.get("schema_version") != expected_schema
                or frame.get("protocol_version") != expected_schema):
            raise ValueError(
                f"raw frame schema/protocol must both be {expected_schema}")
        if expected_schema in (3, 4):
            expected_identity = (
                ("bridge-v3-20261009", "workledger-v3-20261009")
                if expected_schema == 3 else
                ("bridge-v4-20261011", "workledger-v4-20261011"))
            if (frame.get("bridge_protocol"),
                    frame.get("work_ledger_schema")) != expected_identity:
                raise ValueError(
                    f"raw v{expected_schema} frame identity strings disagree")
        frames.append(entry)
    return frames


def _delivery_envelopes(raw_result: dict, *, expected_schema: int) -> list[dict]:
    """Read the fixture's bounded all-delivery frame channel, not audit tags."""
    frames = raw_result.get("delivery_frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("raw fixture delivery_frames must be a nonempty array")
    certificate = raw_result.get("terminal_certificate")
    if (not isinstance(certificate, dict)
            or certificate.get("final_delivery_sequence") != len(frames) - 1):
        raise ValueError(
            "delivery_frames do not cover terminal_certificate's final sequence")
    for sequence, frame in enumerate(frames):
        if not isinstance(frame, dict):
            raise ValueError("delivery frame envelope must be an object")
        if (frame.get("delivery_sequence") != sequence
                or type(frame.get("schema_version")) is not int
                or type(frame.get("protocol_version")) is not int
                or frame.get("schema_version") != expected_schema
                or frame.get("protocol_version") != expected_schema):
            raise ValueError(
                f"delivery frame {sequence} has a sequence/schema mismatch")
        if expected_schema in (3, 4):
            expected_identity = (
                ("bridge-v3-20261009", "workledger-v3-20261009")
                if expected_schema == 3 else
                ("bridge-v4-20261011", "workledger-v4-20261011"))
            if (frame.get("bridge_protocol"),
                    frame.get("work_ledger_schema")) != expected_identity:
                raise ValueError(
                    f"delivery frame {sequence} has invalid v{expected_schema} identity")
        if frame.get("snapshot_kind") != ("full" if sequence == 0 else "delta"):
            raise ValueError(
                f"delivery frame {sequence} has an invalid snapshot kind")
    receipt_sequences = sorted(
        int(row["delivery_sequence"])
        for row in raw_result.get("audit_records", [])
        if isinstance(row, dict) and row.get("kind") == "v2_frame_published"
    )
    if receipt_sequences != list(range(len(frames))):
        raise ValueError(
            "published-frame receipts and delivery_frames cover different sequences")
    focused = _raw_frames(raw_result, expected_schema=expected_schema)
    for entry in focused:
        sequence = int(entry["frame"]["delivery_sequence"])
        if entry["frame"] != frames[sequence]:
            raise ValueError(
                f"focused audit frame {sequence} differs from delivered envelope")
    return frames


def _has_request_close(frame: dict) -> bool:
    return any(run.get("transition_kind") == "request_close"
               for run in frame.get("pipeline_transition_runs", []))


def _ledger_snapshot(harness, *, frame_index: int) -> dict:
    from joint.acceptance_projection import _jsonify
    return {
        "frame_index": frame_index,
        "ordinary": harness.joint_work_ledger.export_state(),
        "pipeline": _jsonify(harness.joint_pipeline_ledger.remaining_view()),
        "ordinary_version": int(harness.joint_work_ledger.version),
        "ordinary_watermark": int(harness.joint_work_ledger.event_watermark),
    }


def _offline_protocol_version(frames: list[dict]) -> int:
    import joint.acceptance_projection as ap

    return ap._retained_frame_protocol_version(frames)


def _normalize_v2_reference_view(value: Any) -> Any:
    """Explicit pre-close v2→current-V4 cursor normalization for comparison.

    R_current has no administrative cancellation counter. A v2 reference
    prefix before any request_close represents zero cancellations, so add the
    current counter field with value 0 only to its cursor-run view. This adapter is never
    imported by a production consumer and is forbidden on closed prefixes.
    """
    if isinstance(value, list):
        return [_normalize_v2_reference_view(item) for item in value]
    if not isinstance(value, dict):
        return value
    result = {key: _normalize_v2_reference_view(item)
              for key, item in value.items()}
    if ("completed_endpoint_count" in result
            and "remaining_endpoint_count" in result
            and "cancelled_endpoint_count" not in result):
        result["cancelled_endpoint_count"] = 0
    return result


def _first_json_difference(reference: Any, current: Any,
                           path: str = "$") -> dict | None:
    """Return one complete leaf/row mismatch for a small review artifact."""
    if type(reference) is not type(current):
        return {"path": path, "kind": "type",
                "reference": type(reference).__name__,
                "current": type(current).__name__}
    if isinstance(reference, dict):
        for key in sorted(set(reference) | set(current)):
            child = f"{path}.{key}"
            if key not in reference:
                return {"path": child, "kind": "missing_reference",
                        "reference": None, "current": current[key]}
            if key not in current:
                return {"path": child, "kind": "missing_current",
                        "reference": reference[key], "current": None}
            difference = _first_json_difference(
                reference[key], current[key], child)
            if difference is not None:
                return difference
        return None
    if isinstance(reference, list):
        for index, (old_item, new_item) in enumerate(
                zip(reference, current)):
            difference = _first_json_difference(
                old_item, new_item, f"{path}[{index}]")
            if difference is not None:
                return difference
        if len(reference) != len(current):
            index = min(len(reference), len(current))
            return {
                "path": f"{path}[{index}]",
                "kind": "length",
                "reference_length": len(reference),
                "current_length": len(current),
                "reference": (reference[index]
                              if index < len(reference) else None),
                "current": (current[index]
                            if index < len(current) else None),
            }
        return None
    if reference != current:
        return {"path": path, "kind": "value",
                "reference": reference, "current": current}
    return None


def _complete_nullable_descriptor_fields(value: Any, typed: Any) -> Any:
    """Complete only omitted null defaults in the v2 descriptor-state copy."""
    if isinstance(value, dict) and isinstance(typed, dict):
        result = {
            key: _complete_nullable_descriptor_fields(item, typed[key])
            if key in typed else copy.deepcopy(item)
            for key, item in value.items()
        }
        for key, item in typed.items():
            if key not in value and item is None:
                result[key] = None
        return result
    if isinstance(value, list) and isinstance(typed, list):
        if len(value) != len(typed):
            return copy.deepcopy(value)
        return [_complete_nullable_descriptor_fields(old, new)
                for old, new in zip(value, typed)]
    return copy.deepcopy(value)


def _normalize_preclose_snapshot_pair(
        reference: dict, current: dict, *, preclose: bool) -> tuple[
            dict, dict, list[dict]]:
    """Normalize only known v2/current-V4 representation differences.

    Descriptor declarations remain compared in full at ``pipeline.descriptors``.
    Before removing the v2 state's redundant declaration/identity copy, verify
    that it denotes that exact typed declaration.  Obligation ordering is
    canonicalized by its key only; every row and value remains in the compare.
    """
    old = copy.deepcopy(reference)
    new = copy.deepcopy(current)
    old_pipeline = old["pipeline"]
    new_pipeline = new["pipeline"]
    errors: list[dict] = []

    old_descriptors = old_pipeline.get("descriptors", {})
    new_descriptors = new_pipeline.get("descriptors", {})
    for side, pipeline, declarations in (
            ("reference", old_pipeline, old_descriptors),
            ("current", new_pipeline, new_descriptors)):
        for segment, state in pipeline.get("descriptor_state", {}).items():
            declaration = declarations.get(segment)
            if "descriptor" in state:
                if declaration is None:
                    errors.append({
                        "path": f"pipeline.descriptor_state.{segment}.descriptor",
                        "kind": "missing_typed_declaration",
                        "side": side,
                    })
                else:
                    completed = _complete_nullable_descriptor_fields(
                        state["descriptor"], declaration)
                    difference = _first_json_difference(
                        completed, declaration,
                        f"pipeline.descriptor_state.{segment}.descriptor")
                    if difference is not None:
                        errors.append(difference)
                state.pop("descriptor")
            if "identity" in state:
                expected_identity = (
                    declaration.get("identity")
                    if isinstance(declaration, dict) else None)
                difference = _first_json_difference(
                    state["identity"], expected_identity,
                    f"pipeline.descriptor_state.{segment}.identity")
                if difference is not None:
                    errors.append(difference)
                state.pop("identity")

    for segment in set(old_pipeline.get("descriptor_state", {})) & set(
            new_pipeline.get("descriptor_state", {})):
        old_state = old_pipeline["descriptor_state"][segment]
        new_state = new_pipeline["descriptor_state"][segment]
        if "request_closed" not in old_state:
            if preclose and new_state.get("request_closed") is False:
                old_state["request_closed"] = False
            else:
                errors.append({
                    "path": f"pipeline.descriptor_state.{segment}.request_closed",
                    "kind": "legacy_default_not_proven",
                    "reference": "absent",
                    "current": new_state.get("request_closed"),
                })

    for pipeline in (old_pipeline, new_pipeline):
        obligations = pipeline.get("obligations", [])
        obligation_ids = [row.get("obligation_id") for row in obligations
                          if isinstance(row, dict)]
        valid_ids = (len(obligation_ids) == len(obligations)
                     and all(isinstance(value, str)
                             for value in obligation_ids))
        if (not valid_ids
                or len(set(obligation_ids)) != len(obligation_ids)):
            errors.append({
                "path": "pipeline.obligations",
                "kind": "invalid_obligation_identity_set",
            })
        else:
            pipeline["obligations"] = sorted(
                obligations, key=lambda row: row["obligation_id"])
    return old, new, errors


def _assert_pending_obligations_and_acks(
        focused_frames: list[dict], delivery_frames: list[dict]) -> dict:
    pending_obligations = False
    pending_resources = False
    pending_acks = False
    ack_done_counts = {"fill_ack_done": 0, "release_ack_done": 0}
    expected_ack_done_counts = {"fill_ack_done": 0, "release_ack_done": 0}
    descriptor_count = 0
    for entry in focused_frames:
        frame = entry["frame"]
        for run in frame.get("pipeline_transition_runs", []):
            for patch in run.get("obligation_patches", []):
                after = patch.get("after")
                if after is not None and int(after.get(
                        "remaining_endpoint_count", 0)) > 0:
                    pending_obligations = True
            for patch in run.get("credit_patches", []):
                after = patch.get("after")
                if after is not None and int(after.get(
                        "outstanding_ack_count", 0)) > 0:
                    pending_acks = True
        for resource in frame.get("active_resources", []):
            if resource.get("resource_type") in (
                    "noc_flow", "hbm_job", "pool_job"):
                remaining = resource.get("remaining", 0)
                if isinstance(remaining, dict):
                    amounts = (remaining.get("ops", 0),
                               remaining.get("bytes", 0),
                               remaining.get("latency_ns", 0))
                    if any((type(value) is int and value > 0)
                           or (type(value) is float
                               and math.isfinite(value) and value > 0.0)
                           for value in amounts):
                        pending_resources = True
                elif ((type(remaining) is int and remaining > 0)
                      or (type(remaining) is float
                          and math.isfinite(remaining) and remaining > 0.0)):
                    pending_resources = True
    for frame in delivery_frames:
        for commit in frame.get("descriptor_commits", []):
            descriptor = commit.get("descriptor")
            if not isinstance(descriptor, dict):
                raise AssertionError(
                    "delivered descriptor commit has no descriptor object")
            descriptor_count += 1
            for endpoint_run in descriptor.get("endpoint_runs", []):
                tile_count = endpoint_run.get("tile_count")
                shard_ordinals = endpoint_run.get("shard_ordinals")
                if (type(tile_count) is not int or tile_count < 1
                        or not isinstance(shard_ordinals, list)
                        or not shard_ordinals):
                    raise AssertionError(
                        "descriptor endpoint run lacks bounded tile/shard counts")
                leg_count = tile_count * len(shard_ordinals)
                for leg in endpoint_run.get("legs", []):
                    endpoint = leg.get("endpoint")
                    if endpoint in ("fill_ack_send", "fill_ack_recv"):
                        expected_ack_done_counts["fill_ack_done"] += leg_count
                    elif endpoint in ("release_ack_send", "release_ack_recv"):
                        expected_ack_done_counts["release_ack_done"] += leg_count
        for run in frame.get("pipeline_transition_runs", []):
            kind = run.get("transition_kind")
            if kind in ack_done_counts:
                ack_done_counts[kind] += int(run.get("repeat_count", 1))
            for patch in run.get("credit_patches", []):
                after = patch.get("after")
                if after is not None and int(after.get(
                        "outstanding_ack_count", 0)) > 0:
                    pending_acks = True
    if not pending_obligations:
        raise AssertionError("no current V4 prefix exposed an active endpoint obligation")
    if not pending_resources:
        raise AssertionError("no focused V4 frame exposed active resource work")
    if not pending_acks:
        raise AssertionError(
            "no delivered V4 transition exposed an outstanding release ACK")
    if descriptor_count == 0 or any(
            count == 0 for count in expected_ack_done_counts.values()):
        raise AssertionError(
            "full delivery tape has no descriptor-declared fill and release ACK legs")
    if ack_done_counts != expected_ack_done_counts:
        raise AssertionError(
            "completed ACK legs must match the delivered descriptor endpoint runs; "
            f"expected {expected_ack_done_counts}, observed {ack_done_counts}")
    return {"pending_obligations": pending_obligations,
            "pending_resources": pending_resources,
            "pending_release_ack_observed": pending_acks,
            "descriptor_commit_count": descriptor_count,
            "descriptor_ack_leg_counts": expected_ack_done_counts,
            "ack_done_transition_counts": ack_done_counts,
            "ack_done_transition_total": sum(ack_done_counts.values())}


def _verify_raw_versions_and_shape(
        v2_raw: dict, v4_raw: dict) -> tuple[list, list, list, dict]:
    old_frames = _raw_frames(v2_raw, expected_schema=2)
    new_frames = _raw_frames(v4_raw, expected_schema=4)
    delivery_frames = _delivery_envelopes(v4_raw, expected_schema=4)
    if len(old_frames) < 3 or len(new_frames) < 3:
        raise AssertionError("paired fixture results need initial + two delta frames")
    for frames, label in ((old_frames, "R_current v2"), (new_frames, "current V4")):
        snapshots = [row["frame"].get("snapshot_kind") for row in frames]
        if snapshots[0] != "full" or snapshots.count("delta") < 2:
            raise AssertionError(
                f"{label} must have an initial full frame and >=2 deltas: {snapshots}")
    ack_evidence = _assert_pending_obligations_and_acks(
        new_frames, delivery_frames)
    return old_frames, new_frames, delivery_frames, ack_evidence


def _reference_worker_source() -> str:
    # This embedded worker loads only the caller-supplied frozen reference
    # root. Its decoder calls are not the current R_work V4 offline harness.
    return r'''
import copy, importlib.util, json, sys, tempfile
from pathlib import Path
reference_root, fixture_dir, raw_path, output_path, library_path = sys.argv[1:6]
workload = Path(reference_root) / "sh_test_mesh" / "workload" / "llama2_7b_inference"
driver_path = Path(reference_root) / "sh_test_mesh" / "run_scripts" / "joint_accel_acceptance.py"
sys.path.insert(0, str(workload))
spec = importlib.util.spec_from_file_location("reference_driver", driver_path)
driver = importlib.util.module_from_spec(spec); spec.loader.exec_module(driver)
import joint.acceptance_projection as ap
from joint.prediction_config import JointPredictionConfig, PredictionParams
raw = json.loads(Path(raw_path).read_text(encoding="utf-8"))
case = Path(fixture_dir)
state_path = case / "state.json"
state = driver.validate_fixture_state(json.loads(state_path.read_text(encoding="utf-8")), state_path)
frames = [row for row in raw.get("audit_records", []) if isinstance(row, dict) and row.get("kind") == "raw_v2_frame"]
prefix = None
failure = None
for count in range(len(frames), 1, -1):
    candidate = copy.deepcopy(raw)
    candidate["audit_records"] = [row for row in candidate.get("audit_records", [])
                                 if row.get("kind") != "raw_v2_frame"] + frames[:count]
    candidate["boundary_records"] = candidate.get("boundary_records", [])[:count]
    try:
        prefix = ap.canonicalize_raw_frames(candidate, state, source_root=case)
        break
    except Exception as exc:
        failure = f"{type(exc).__name__}: {exc}"
if prefix is None:
    # Retain the actual old model's valid apply prefix even if its planner
    # adapter rejects a later post-close settle boundary.
    harness = ap._build_dual_ledger_harness()
    rows = []
    for index, entry in enumerate(frames):
        try:
            harness._apply_work_ledger({"work_ledger": entry["frame"]})
        except Exception as exc:
            failure = f"{type(exc).__name__}: {exc}"
            break
        rows.append({
            "frame_index": index,
            "ordinary": harness.joint_work_ledger.export_state(),
            "pipeline": ap._jsonify(harness.joint_pipeline_ledger.remaining_view()),
            "ordinary_version": int(harness.joint_work_ledger.version),
            "ordinary_watermark": int(harness.joint_work_ledger.event_watermark),
        })
    output = {"ledger_rows": rows, "canonicalized": None,
              "boundary_count": len(rows), "failure": failure}
else:
    harness = ap._build_dual_ledger_harness()
    rows = []
    for index, entry in enumerate(frames[:len(prefix["boundary_records"])]):
        harness._apply_work_ledger({"work_ledger": entry["frame"]})
        rows.append({
            "frame_index": index,
            "ordinary": harness.joint_work_ledger.export_state(),
            "pipeline": ap._jsonify(harness.joint_pipeline_ledger.remaining_view()),
            "ordinary_version": int(harness.joint_work_ledger.version),
            "ordinary_watermark": int(harness.joint_work_ledger.event_watermark),
        })
    params = JointPredictionConfig(params=PredictionParams(
        projection_event_budget=4194304, window_tile_limit=1,
        window_local_error_ppm=10000, window_refine_margin_ppm=20000))
    evals = []
    metadata_sources = []
    world_source_prefixes = []
    eval_boundaries = prefix["boundary_records"][:8]
    if len(frames[:len(eval_boundaries)]) != len(eval_boundaries):
        raise RuntimeError("reference raw frame prefix is shorter than its boundaries")
    asset = ap._resolve_runtime_asset_strict(state)
    with tempfile.TemporaryDirectory(prefix="joint-reference-prefix-source-") as runtime_dir:
        runtime = ap.hydrate_runtime_asset(asset, private_dir=runtime_dir)
        from online.sh30_online_scheduler import Sh30OnlineScheduler
        from joint.action_work_graph import LedgerWorldSource
        from joint.runtime_snapshot import build_decision_snapshot

        link_shell = Sh30OnlineScheduler.__new__(Sh30OnlineScheduler)
        link_shell.hardware = runtime.hardware
        link_shell._telemetry_link_id_map = None
        link_mapper = link_shell._telemetry_endpoint_link_key
        source_harness = ap._build_dual_ledger_harness()
        for index, boundary in enumerate(eval_boundaries):
            frame = frames[index]["frame"]
            source_harness._apply_work_ledger({"work_ledger": frame})
            work_ledger = source_harness.joint_work_ledger
            pipeline_ledger = source_harness.joint_pipeline_ledger
            if int(pipeline_ledger.delivery_sequence) != int(frame["delivery_sequence"]):
                raise RuntimeError("reference prefix pipeline frontier missed its raw frame")
            snapshot = build_decision_snapshot({
                "delivery_sequence": int(pipeline_ledger.delivery_sequence),
                "tick": int(boundary["tick_ns"]),
                "arrivals": [],
            }, versions=None, ledger_state=work_ledger.snapshot_state(),
                ledger_projection=work_ledger.projection_snapshot(),
                pipeline_ledger_state=pipeline_ledger.snapshot_state())
            source = LedgerWorldSource(
                ledger=work_ledger, snapshot=snapshot,
                link_id_map_fn=link_mapper)
            eval_boundary = copy.deepcopy(boundary)
            metadata = eval_boundary.get("causal_metadata")
            planner = eval_boundary.get("planner_state")
            if not isinstance(metadata, dict) or not isinstance(planner, dict):
                raise RuntimeError("reference boundary lacks causal planner recipe")
            context = metadata.get("context_tokens")
            horizon = metadata.get("horizon_tokens")
            if (type(context) is not int or context < 1
                    or type(horizon) is not int or horizon < 1):
                raise RuntimeError("reference causal context/horizon is invalid")
            recipe = planner.get("forecast_generator")
            expected_recipe = ap.export_forecast_generator(
                context_tokens=context, horizon_tokens=horizon)
            if recipe != expected_recipe:
                raise RuntimeError(
                    "reference planner forecast recipe does not match its "
                    "causal context/horizon/hash")
            existing = metadata.get("forecast_generator")
            if existing is not None and existing != recipe:
                raise RuntimeError(
                    "reference causal metadata conflicts with planner recipe")
            metadata["forecast_generator"] = copy.deepcopy(recipe)
            metadata_sources.append({
                "context_tokens": context,
                "horizon_tokens": horizon,
                "identity": recipe["identity"],
                "semantic_sha256": recipe["semantic_sha256"],
                "source": "same_boundary.planner_state.forecast_generator",
            })
            causal_work = eval_boundary["causal_prestate"]["work_ledger"]
            causal_work["_prebuilt_ledger_world_source"] = source
            pipeline_view = causal_work.get("pipeline", {})
            active_resources = snapshot.pipeline_ledger_state.get(
                "active_resources", ())
            obligations = pipeline_view.get("obligations", ())
            stage_input = planner.get("stage_input", {})
            runtime_state = stage_input.get("runtime_state", {})
            world_source_prefixes.append({
                "boundary_ordinal": int(boundary["boundary_ordinal"]),
                "stage": boundary["stage"],
                "delivery_sequence": int(frame["delivery_sequence"]),
                "tick_ns": int(boundary["tick_ns"]),
                "ordinary_pricing_row_count": len(
                    snapshot.ledger_projection.pricing_rows),
                "pipeline_active_resource_count": len(active_resources),
                "pipeline_resource_types": {
                    kind: sum(1 for row in active_resources
                              if row.get("resource_type") == kind)
                    for kind in sorted({str(row.get("resource_type", "<missing>"))
                                        for row in active_resources})
                },
                "pipeline_obligation_count": len(obligations),
                "planner_obligation_count": len(runtime_state.get("obligations", ())),
            })
            evals.append(ap.evaluate_boundary(
                eval_boundary, params, library_path=library_path,
                fixture_state=state))
    output = {"ledger_rows": rows, "canonicalized": prefix,
              "boundary_count": len(prefix["boundary_records"]),
              "evaluations": evals,
              "causal_metadata_completion": metadata_sources,
              "world_source_prefixes": world_source_prefixes,
              "failure": failure}
Path(output_path).write_text(json.dumps(output, sort_keys=True), encoding="utf-8")
'''


def _run_reference_worker(reference_root: Path, fixture_dir: Path,
                          raw_path: Path, library_path: Path,
                          output_path: Path) -> dict:
    argv = [sys.executable, "-c", _reference_worker_source(),
            str(reference_root), str(fixture_dir), str(raw_path),
            str(output_path), str(library_path)]
    completed = subprocess.run(argv, check=False, capture_output=True,
                               text=True, timeout=30)
    if completed.returncode != 0:
        raise RuntimeError(
            "R_current reference worker failed: "
            f"exit={completed.returncode}, stderr={completed.stderr[-3000:]}")
    return json.loads(output_path.read_text(encoding="utf-8"))


def _verify_bad_input_atomicity(delivery_frames: list[dict]) -> None:
    import joint.acceptance_projection as ap

    protocol_version = _offline_protocol_version(delivery_frames)
    harness = ap._build_dual_ledger_harness(protocol_version)
    harness._apply_work_ledger({"work_ledger": delivery_frames[0]})
    candidate = None
    for frame_index, frame in enumerate(delivery_frames[1:], start=1):
        for run_index, run in enumerate(frame.get("pipeline_transition_runs", [])):
            for patch_index, patch in enumerate(run.get("slot_patches", [])):
                if protocol_version == 4:
                    if not (isinstance(patch, dict)
                            and patch.get("form") == "target_delta"):
                        continue
                    operation_index = next((index for index, operation in
                        enumerate(patch.get("operations", []))
                        if operation.get("op") == "consume"), None)
                    candidate = (frame_index, frame, run_index, patch_index,
                                 operation_index)
                    break
                row = patch.get("after") or patch.get("before")
                if row is not None:
                    candidate = (frame_index, frame, run_index, patch_index, None)
                    break
            if candidate:
                break
        if candidate:
            break
    if candidate is None:
        raise AssertionError(
            f"V{protocol_version} integration frames contain no slot patch to corrupt")
    frame_index, frame, run_index, patch_index, operation_index = candidate
    for frame in delivery_frames[1:frame_index]:
        harness._apply_work_ledger({"work_ledger": frame})
    corrupt = copy.deepcopy(frame)
    patch = corrupt["pipeline_transition_runs"][run_index]["slot_patches"][patch_index]
    if protocol_version == 4:
        if operation_index is not None:
            operation = patch["operations"][operation_index]
            # Keep the V4 DTO well-typed and sorted while falsifying the
            # descriptor-derived cause result.
            operation["add_shards"] = [(1 << 64) - 1]
        else:
            patch["before_slot_generation"] += 1
    else:
        row = patch.get("after") or patch.get("before")
        row["role"] = "__invalid_sparse_test_role__"
    before = _ledger_snapshot(harness, frame_index=frame_index - 1)
    try:
        harness._apply_work_ledger({"work_ledger": corrupt})
    except Exception:
        pass
    else:
        raise AssertionError(f"malformed v{protocol_version} slot patch was accepted")
    after = _ledger_snapshot(harness, frame_index=frame_index - 1)
    if before != after:
        raise AssertionError(
            f"bad V{protocol_version} input mutated a ledger before rejection")


def _write_bridge_journal(bridge_dir: Path, frames: list[dict]) -> None:
    from online.decision_bridge import BridgeServer
    from joint.acceptance_projection import _build_dual_ledger_harness
    bridge_dir.mkdir(parents=True, exist_ok=True)
    # Capture the exact canonical child bytes returned by the production
    # dual-ledger consumer for each accepted frame. BridgeServer's journal
    # callback below reads these bytes, matching the scheduler cache contract.
    protocol_version = _offline_protocol_version(frames)
    snapshot_handle_kind = ("work_ledger_v3" if protocol_version == 3
                            else "work_ledger_v4")
    ledger_harness = _build_dual_ledger_harness(protocol_version)
    canonical_by_sequence = {}
    for frame in frames:
        sequence = int(frame["delivery_sequence"])
        canonical = ledger_harness._apply_work_ledger(
            {"work_ledger": frame})
        if not isinstance(canonical, bytes):
            raise AssertionError(
                "production ledger consumer did not return canonical bytes")
        canonical_by_sequence[sequence] = canonical
    note_by_sequence = {
        int(frame["delivery_sequence"]): {
            "snapshot_kind": frame.get("snapshot_kind"),
            "event_watermark": None,
        }
        for frame in frames
    }
    server = BridgeServer(
        str(bridge_dir), canonical_request_producer=True,
        work_ledger_bytes_source=lambda sequence: (
            canonical_by_sequence[sequence],
            note_by_sequence[sequence]))
    try:
        for frame in frames:
            sequence = int(frame["delivery_sequence"])
            request = {
                "schema_version": 2,
                "delivery_sequence": sequence,
                "tick": int(frame["tick"]),
                "snapshot_handle": {
                    "kind": snapshot_handle_kind,
                    "epoch": frame["epoch"],
                    "tick": int(frame["tick"]),
                },
                "work_ledger": frame,
            }
            (bridge_dir / f"request_{sequence}.json").write_text(
                json.dumps(request, separators=(",", ":")),
                encoding="utf-8")
            server._handle_request(
                sequence,
                lambda _request, _segment: {"pipeline_descriptors": [],
                                            "kernel_contract": {}},
            )
    finally:
        server._close_audit_streams()


def _journal_reader_worker_source() -> str:
    return r'''
import json, sys
from pathlib import Path
workload, bridge_dir, output_path = sys.argv[1:4]
sys.path.insert(0, workload)
from online.bridge_request_journal import iter_request_records
import joint.acceptance_projection as ap
records = list(iter_request_records(bridge_dir, attach_work_ledger=True))
if len(records) < 3:
    raise RuntimeError("journal roundtrip expected full frame + two deltas")
protocol = ap._retained_frame_protocol_version(
    [request["work_ledger"] for _sequence, request in records])
harness = ap._build_dual_ledger_harness(protocol)
expected_handle_kind = "work_ledger_v3" if protocol == 3 else "work_ledger_v4"
for sequence, request in records:
    if request.get("snapshot_handle", {}).get("kind") != expected_handle_kind:
        raise RuntimeError("journal lost the protocol-specific snapshot discriminator")
    harness._apply_work_ledger(request)
state = {
    "ordinary": harness.joint_work_ledger.export_state(),
    "pipeline": ap._jsonify(harness.joint_pipeline_ledger.remaining_view()),
    "version": int(harness.joint_work_ledger.version),
    "watermark": int(harness.joint_work_ledger.event_watermark),
}
Path(output_path).write_text(json.dumps(state, sort_keys=True), encoding="utf-8")
'''


def _verify_journal_new_process(frames: list[dict], expected: dict) -> None:
    from online.bridge_request_journal import iter_request_records

    protocol_version = _offline_protocol_version(frames)
    with tempfile.TemporaryDirectory(
            prefix=f"joint-v{protocol_version}-journal-") as temp:
        root = Path(temp)
        journal_dir = root / "bridge"
        _write_bridge_journal(journal_dir, frames)
        # The reader runs in a clean interpreter and applies every attached
        # delivery, including the last delta after reconstructing its prefix.
        output_path = root / "fresh-process-state.json"
        completed = subprocess.run(
            [sys.executable, "-c", _journal_reader_worker_source(),
             str(WORKLOAD_DIR), str(journal_dir), str(output_path)],
            check=False, capture_output=True, text=True, timeout=30)
        if completed.returncode != 0:
            raise RuntimeError(
                "new-process journal replay failed: "
                f"exit={completed.returncode}, stderr={completed.stderr[-3000:]}")
        actual = json.loads(output_path.read_text(encoding="utf-8"))
        expected_state = {
            "ordinary": expected["ordinary"],
            "pipeline": expected["pipeline"],
            "version": expected["ordinary_version"],
            "watermark": expected["ordinary_watermark"],
        }
        if actual != expected_state:
            raise AssertionError("fresh-process journal replay differs from direct apply")

        # Truncated journal JSON and truncated gzip sidecar each fail closed.
        journal_copy = root / "bad-journal"
        shutil.copytree(journal_dir, journal_copy)
        journal_path = journal_copy / REQUEST_JOURNAL_NAME
        journal_path.write_bytes(journal_path.read_bytes()[:-5])
        try:
            list(iter_request_records(journal_copy, attach_work_ledger=False))
        except Exception:
            pass
        else:
            raise AssertionError("truncated request journal was accepted")

        sidecar_copy = root / "bad-sidecar"
        shutil.copytree(journal_dir, sidecar_copy)
        sidecar_path = sidecar_copy / WORK_LEDGER_SIDECAR_NAME
        payload = sidecar_path.read_bytes()
        sidecar_path.write_bytes(payload[:max(1, len(payload) // 2)])
        try:
            list(iter_request_records(sidecar_copy, attach_work_ledger=True))
        except Exception:
            pass
        else:
            raise AssertionError("truncated work-ledger sidecar was accepted")


def _proof_prediction_config():
    from joint.prediction_config import JointPredictionConfig, PredictionParams
    return JointPredictionConfig(params=PredictionParams(
        projection_event_budget=4_194_304,
        window_tile_limit=1,
        window_local_error_ppm=10_000,
        window_refine_margin_ppm=20_000,
    ))


def _complete_causal_generator_metadata(boundary: dict, ap) -> dict:
    """Copy the same-boundary planner recipe into causal metadata, test-only.

    Stage-B cold canonicalization currently writes context/horizon into
    causal_metadata and independently writes the complete forecast recipe into
    planner_state.  The projection contract needs that same recipe in both
    places.  This helper permits only the existing no-exogenous recipe and
    verifies its dimensions and semantic digest before copying it; it cannot
    add future execution facts or construct a new forecast.
    """
    metadata = boundary.get("causal_metadata")
    planner_state = boundary.get("planner_state")
    if not isinstance(metadata, dict) or not isinstance(planner_state, dict):
        raise AssertionError("boundary lacks causal metadata or planner state")
    context = metadata.get("context_tokens")
    horizon = metadata.get("horizon_tokens")
    if (isinstance(context, bool) or not isinstance(context, int)
            or context < 1 or isinstance(horizon, bool)
            or not isinstance(horizon, int) or horizon < 1):
        raise AssertionError(
            "causal metadata context/horizon must be positive integer facts")
    generator = planner_state.get("forecast_generator")
    if not isinstance(generator, dict) or set(generator) != {
            "identity", "causal_inputs", "semantic_sha256"}:
        raise AssertionError(
            "same-boundary planner state lacks the typed forecast recipe")
    if generator.get("identity") != ap.NO_EXOGENOUS_GENERATOR_IDENTITY:
        raise AssertionError(
            "micro fixture requires its existing no-exogenous generator")
    causal_inputs = generator.get("causal_inputs")
    if not isinstance(causal_inputs, dict) or set(causal_inputs) != {
            "context_tokens", "horizon_tokens", "generator_inputs"}:
        raise AssertionError(
            "same-boundary forecast recipe has unexpected causal inputs")
    recipe_context = causal_inputs.get("context_tokens")
    recipe_horizon = causal_inputs.get("horizon_tokens")
    generator_inputs = causal_inputs.get("generator_inputs")
    if (isinstance(recipe_context, bool)
            or not isinstance(recipe_context, int)
            or isinstance(recipe_horizon, bool)
            or not isinstance(recipe_horizon, int)
            or recipe_context != context or recipe_horizon != horizon
            or generator_inputs != {}):
        raise AssertionError(
            "same-boundary forecast recipe disagrees with causal context, "
            "horizon, or no-exogenous input contract")
    expected_sha = ap._semantic_sha256({
        "identity": generator["identity"],
        "causal_inputs": causal_inputs,
    })
    if generator.get("semantic_sha256") != expected_sha:
        raise AssertionError(
            "same-boundary forecast recipe semantic_sha256 is invalid")
    if generator != ap.export_forecast_generator(
            context_tokens=context, horizon_tokens=horizon):
        raise AssertionError(
            "same-boundary forecast recipe differs from the canonical "
            "no-exogenous export for its causal dimensions")
    existing = metadata.get("forecast_generator")
    if existing is not None and existing != generator:
        raise AssertionError(
            "causal metadata generator conflicts with planner recipe")
    metadata["forecast_generator"] = copy.deepcopy(generator)
    return {
        "source": "same_boundary.planner_state.forecast_generator",
        "context_tokens": context,
        "horizon_tokens": horizon,
        "identity": generator["identity"],
        "semantic_sha256": expected_sha,
    }


def _t2_dto_diagnostic(side: str, index: int, dto: dict, ap) -> dict:
    candidates = dto.get("candidates") if isinstance(dto, dict) else None
    inapplicable = (dto.get("inapplicable_candidates")
                    if isinstance(dto, dict) else None)
    valid = (isinstance(dto, dict) and dto.get("status") == "PASS"
             and isinstance(candidates, list)
             and isinstance(inapplicable, list))
    row = {
        "side": side,
        "boundary_index": index,
        "stage": dto.get("stage") if isinstance(dto, dict) else None,
        "request_id": dto.get("request_id") if isinstance(dto, dict) else None,
        "boundary_ordinal": (dto.get("boundary_ordinal")
                             if isinstance(dto, dict) else None),
        "tick_ns": dto.get("tick_ns") if isinstance(dto, dict) else None,
        "version": dto.get("version") if isinstance(dto, dict) else None,
        "status": dto.get("status") if isinstance(dto, dict) else None,
        "reason": dto.get("reason") if isinstance(dto, dict) else (
            "evaluation DTO is not an object"),
        "candidate_fields_present": isinstance(candidates, list),
        "inapplicable_candidate_fields_present": isinstance(inapplicable, list),
        "valid_candidate_dto": valid,
        "candidate_count": len(candidates) if isinstance(candidates, list) else None,
        "inapplicable_candidate_count": (
            len(inapplicable) if isinstance(inapplicable, list) else None),
    }
    if not valid:
        row["dto"] = ap._jsonify(dto)
    return row


def _require_t2_candidate_dto(side: str, index: int, dto: dict,
                              diagnostic_path: Path) -> None:
    if (not isinstance(dto, dict) or dto.get("status") != "PASS"
            or not isinstance(dto.get("candidates"), list)
            or not isinstance(dto.get("inapplicable_candidates"), list)):
        status = dto.get("status") if isinstance(dto, dict) else None
        reason = dto.get("reason") if isinstance(dto, dict) else (
            "evaluation DTO is not an object")
        raise AssertionError(
            f"{side} T2 boundary {index} did not produce a complete "
            f"candidate DTO: status={status!r}, reason={reason!r}; "
            f"details: {diagnostic_path}")


def _verified_test_link_mapper(runtime):
    """Use the source scheduler's C++ LinkId → mesh endpoint conversion."""
    from online.sh30_online_scheduler import Sh30OnlineScheduler

    shell = Sh30OnlineScheduler.__new__(Sh30OnlineScheduler)
    shell.hardware = runtime.hardware
    shell._telemetry_link_id_map = None
    return shell._telemetry_endpoint_link_key


def _prefix_world_source(harness, frame: dict, boundary: dict, ap,
                         runtime, link_mapper):
    """Build the price source immediately after this boundary's accepted frame.

    Match the source tree's production ledger API: R_work uses its cheap
    pricing snapshots; frozen R_current uses the projection/full-state APIs
    that its online scheduler still exposes.
    """
    from joint.action_work_graph import LedgerWorldSource
    from joint.runtime_snapshot import build_decision_snapshot

    work_ledger = harness.joint_work_ledger
    pipeline_ledger = harness.joint_pipeline_ledger
    work_pricing = getattr(work_ledger, "pricing_snapshot", None)
    if callable(work_pricing):
        ledger_state = {}
        ledger_projection = work_pricing()
        pricing_only = True
    else:
        ledger_state = work_ledger.snapshot_state()
        ledger_projection = work_ledger.projection_snapshot()
        pricing_only = False
    pipeline_pricing = getattr(pipeline_ledger, "pricing_snapshot", None)
    pipeline_state = (
        pipeline_pricing() if callable(pipeline_pricing)
        else pipeline_ledger.snapshot_state())
    delivery_sequence = int(pipeline_ledger.delivery_sequence)
    if delivery_sequence != int(frame["delivery_sequence"]):
        raise AssertionError(
            "prefix pipeline frontier is not at the matching raw frame: "
            f"ledger={delivery_sequence}, frame={frame['delivery_sequence']}")
    tick_ns = int(boundary["tick_ns"])
    delta = {
        "delivery_sequence": delivery_sequence,
        "tick": tick_ns,
        "arrivals": [],
    }
    snapshot_kwargs = {
        "versions": None,
        "ledger_state": ledger_state,
        "ledger_projection": ledger_projection,
        "pipeline_ledger_state": pipeline_state,
    }
    if pricing_only:
        snapshot_kwargs["pricing_only"] = True
    snapshot = build_decision_snapshot(delta, **snapshot_kwargs)
    return LedgerWorldSource(
        ledger=work_ledger,
        snapshot=snapshot,
        link_id_map_fn=link_mapper)


def _world_source_signature(source, ap) -> dict:
    snapshot = source._snapshot
    projection = snapshot.ledger_projection
    pipeline_state = snapshot.pipeline_ledger_state
    if projection is None or pipeline_state is None:
        raise AssertionError(
            "prefix source lacks the ordinary or pipeline pricing frontier")
    return {
        "snapshot": ap._jsonify(snapshot.to_dict()),
        "ordinary_pricing_rows": ap._jsonify(projection.pricing_rows),
        "ordinary_committed_not_issued": int(
            projection.committed_not_issued),
        "pipeline_active_resources": ap._jsonify(
            pipeline_state.get("active_resources", ())),
        "pipeline_state_version": pipeline_state.get("state_version"),
    }


def _prefix_world_source_evidence(boundary: dict, source, ap,
                                  frame: dict) -> dict:
    signature = _world_source_signature(source, ap)
    pipeline = (boundary.get("causal_prestate", {})
                .get("work_ledger", {}).get("pipeline", {}))
    obligations = pipeline.get("obligations", ())
    if isinstance(obligations, dict):
        obligation_count = len(obligations)
    else:
        obligation_count = len(obligations)
    planner_state = boundary.get("planner_state", {})
    stage_input = planner_state.get("stage_input", {})
    runtime_state = stage_input.get("runtime_state", {})
    planner_obligations = runtime_state.get("obligations", ())
    return {
        "boundary_ordinal": int(boundary["boundary_ordinal"]),
        "stage": boundary["stage"],
        "delivery_sequence": int(frame["delivery_sequence"]),
        "tick_ns": int(boundary["tick_ns"]),
        "ordinary_pricing_row_count": len(
            signature["ordinary_pricing_rows"]),
        "pipeline_active_resource_count": len(
            signature["pipeline_active_resources"]),
        "pipeline_obligation_count": obligation_count,
        "planner_obligation_count": len(planner_obligations),
        "pipeline_resource_types": _resource_type_counts(
            signature["pipeline_active_resources"]),
    }


def _resource_type_counts(rows) -> dict:
    counts = {}
    for row in rows:
        kind = str(row.get("resource_type", "<missing>"))
        counts[kind] = counts.get(kind, 0) + 1
    return dict(sorted(counts.items()))


def _build_prefix_world_sources(boundaries: list[dict], entries: list[dict],
                                ap, runtime) -> tuple[list, list[dict]]:
    if len(boundaries) > len(entries):
        raise AssertionError(
            "evaluation boundary prefix is longer than its raw-frame prefix")
    protocol_version = _offline_protocol_version(
        [entry["frame"] for entry in entries])
    harness = ap._build_dual_ledger_harness(protocol_version)
    link_mapper = _verified_test_link_mapper(runtime)
    sources = []
    evidence = []
    for index, (boundary, entry) in enumerate(zip(boundaries, entries)):
        frame = entry["frame"]
        if (entry.get("boundary_ordinal") is not None
                and int(entry["boundary_ordinal"])
                != int(boundary["boundary_ordinal"])):
            raise AssertionError(
                "raw frame and evaluation boundary do not share an ordinal "
                f"at prefix index {index}")
        harness._apply_work_ledger({"work_ledger": frame})
        source = _prefix_world_source(
            harness, frame, boundary, ap, runtime, link_mapper)
        sources.append(source)
        evidence.append(_prefix_world_source_evidence(
            boundary, source, ap, frame))
    return sources, evidence


def _verify_current_replay_fallback(boundary: dict, entries: list[dict],
                                    prebuilt_source, ap, runtime) -> dict:
    if not entries:
        raise AssertionError("current fallback check requires real raw frames")
    probe = copy.deepcopy(boundary)
    work_view = probe["causal_prestate"]["work_ledger"]
    work_view.pop("_prebuilt_ledger_world_source", None)
    work_view["replay_v3_frames"] = [entry["frame"] for entry in entries]
    replayed = ap._hydrate_world_source(probe, runtime)
    if replayed is None:
        raise AssertionError("nonempty current prefix replay became empty")
    expected_signature = _world_source_signature(prebuilt_source, ap)
    replayed_signature = _world_source_signature(replayed, ap)
    difference = _first_json_difference(
        expected_signature, replayed_signature,
        path="same_prefix_world_source")
    if difference is not None:
        raise AssertionError(
            "current retained-frame fallback differs from its same-prefix "
            f"prebuilt world source: {difference}")
    resource_types = _resource_type_counts(
        expected_signature["pipeline_active_resources"])
    if (sum(resource_types.values()) < 8
            or resource_types.get("hbm_job", 0) < 1
            or resource_types.get("noc_flow", 0) < 1):
        raise AssertionError(
            "saved final prefix must carry at least eight real HBM/NoC "
            f"pipeline resources, got {resource_types}")
    return {
        "status": "PASS",
        "replayed_frame_count": len(entries),
        "boundary_ordinal": int(boundary["boundary_ordinal"]),
        "delivery_sequence": int(entries[-1]["frame"]["delivery_sequence"]),
        "resource_count": sum(resource_types.values()),
        "resource_types": resource_types,
        "matching_fields": [
            "snapshot.to_dict (including versions, sequence, watermark, tick)",
            "ordinary pricing rows and committed-not-issued count",
            "pipeline active_resources and state_version",
        ],
    }


def compare_raw_pair(*, fixture_dir: Path, raw_v2_path: Path,
                     raw_v4_path: Path, reference_root: Path,
                     current_library: Path, reference_library: Path,
                     output_path: Path,
                     archive_v3_path: Path | None = None) -> dict:
    import joint.acceptance_projection as ap
    driver = _load_driver()
    state_path = fixture_dir / "state.json"
    state = driver.validate_fixture_state(
        json.loads(state_path.read_text(encoding="utf-8")), state_path)
    v2_raw = json.loads(raw_v2_path.read_text(encoding="utf-8"))
    v4_raw = json.loads(raw_v4_path.read_text(encoding="utf-8"))
    old_frames, new_frames, delivery_frames, ack_evidence = (
        _verify_raw_versions_and_shape(v2_raw, v4_raw))

    v4_canonical = ap.canonicalize_raw_frames(
        v4_raw, state, source_root=fixture_dir)
    archive_v3_summary = None
    if archive_v3_path is not None:
        # Preserve a caller-supplied V3 archive as raw bytes; use only the
        # strict offline V3 decoder and never promote/reserialize it as V4.
        archive_raw = json.loads(
            archive_v3_path.read_text(encoding="utf-8"))
        archive_frames = _raw_frames(archive_raw, expected_schema=3)
        if not archive_frames:
            raise AssertionError("--archive-v3 must retain at least one raw frame")
        if _offline_protocol_version(
                [entry["frame"] for entry in archive_frames]) != 3:
            raise AssertionError("--archive-v3 must contain only V3 archive frames")
        archived_view = ap._replay_ledger_views(
            [entry["frame"] for entry in archive_frames])
        archive_v3_summary = {
            "path": str(archive_v3_path),
            "sha256": _sha256_file(archive_v3_path),
            "frame_count": len(archive_frames),
            "replayed_frame_count": archived_view["replayed_frame_count"],
            "scope": "exact offline V3 archive replay",
        }
    with tempfile.TemporaryDirectory(prefix="joint-v2-reference-") as temp:
        legacy = _run_reference_worker(
            reference_root, fixture_dir, raw_v2_path, reference_library,
            Path(temp) / "reference-output.json")
    reference_rows = legacy.get("ledger_rows", [])
    if legacy.get("canonicalized") is not None:
        reference_records = legacy["canonicalized"]["boundary_records"]
    else:
        reference_records = []
    valid_count = min(len(reference_rows), len(new_frames),
                      len(v4_canonical["boundary_records"]))
    # The known R_current v2 receiver cannot consume administrative close-tail
    # settlement.  Its valid prefix must still include P and one pre-close D.
    if valid_count < 2:
        raise AssertionError(
            "R_current v2 accepted fewer than P + first D pre-close prefixes")

    current_protocol = _offline_protocol_version(
        [entry["frame"] for entry in new_frames])
    current_harness = ap._build_dual_ledger_harness(current_protocol)
    current_rows = []
    for index, entry in enumerate(new_frames[:valid_count]):
        if _has_request_close(entry["frame"]):
            break
        current_harness._apply_work_ledger({"work_ledger": entry["frame"]})
        current_rows.append(_ledger_snapshot(current_harness,
                                             frame_index=index))
    if len(current_rows) < 2:
        raise AssertionError("current V4 frames do not expose two pre-close prefixes")

    reference_rows = reference_rows[:len(current_rows)]
    if any(_has_request_close(entry["frame"])
           for entry in new_frames[:len(current_rows)]):
        raise AssertionError(
            "request_closed default normalization requires a pre-close prefix")
    for index, (reference_row, current_row) in enumerate(
            zip(reference_rows, current_rows)):
        normalized_reference = _normalize_v2_reference_view(reference_row)
        normalized_current = _normalize_v2_reference_view(current_row)
        (normalized_reference, normalized_current,
         normalization_errors) = _normalize_preclose_snapshot_pair(
             normalized_reference, normalized_current, preclose=True)
        difference = _first_json_difference(
            normalized_reference, normalized_current,
            path=f"prefix[{index}]")
        if normalization_errors or difference is not None:
            first_mismatch = (normalization_errors[0]
                              if normalization_errors else difference)
            diff_path = output_path.with_name(
                output_path.name + ".prefix-state-diff.json")
            _write_json(diff_path, {
                "case_id": MICRO_CASE_ID,
                "reference_raw": str(raw_v2_path),
                "current_raw": str(raw_v4_path),
                "prefix_index": index,
                "delivery_sequence": int(
                    new_frames[index]["frame"]["delivery_sequence"]),
                "normalization_errors": normalization_errors,
                "first_mismatch": first_mismatch,
                "normalized_state_difference": difference,
                "comparison_scope": "R_current v2 and current V4 dual-ledger pre-close prefix state",
            })
            raise AssertionError(
                "normalized R_current v2 and current V4 dual-ledger prefix states "
                f"differ at {first_mismatch.get('path', '$')}; details: "
                f"{diff_path}")

    # T2: at most eight identical state snapshots, each evaluates and exports
    # the complete applicable/inapplicable four-action face.
    boundary_records = v4_canonical["boundary_records"][
        :min(valid_count, T2_MAX_BOUNDARIES)]
    reference_evaluations = legacy.get("evaluations", [])
    with tempfile.TemporaryDirectory(prefix="joint-current-prefix-source-") as runtime_dir:
        current_runtime = ap.hydrate_runtime_asset(
            ap._resolve_runtime_asset_strict(state), private_dir=runtime_dir)
        current_world_sources, current_world_source_prefixes = (
            _build_prefix_world_sources(
                boundary_records, new_frames[:len(boundary_records)],
                ap, current_runtime))
        current_replay_fallback = _verify_current_replay_fallback(
            boundary_records[-1], new_frames[:len(boundary_records)],
            current_world_sources[-1], ap, current_runtime)
    dto_diagnostic_path = output_path.with_name(
        output_path.name + ".t2-dto-diagnostics.json")
    dto_diagnostics = {
        "case_id": MICRO_CASE_ID,
        "prediction_config": {
            "projection_event_budget": 4_194_304,
            "window_tile_limit": 1,
            "window_local_error_ppm": 10_000,
            "window_refine_margin_ppm": 20_000,
        },
        "causal_metadata_completion": {
            "reference": legacy.get("causal_metadata_completion", []),
            "current": [],
        },
        "world_source_prefixes": {
            "reference": legacy.get("world_source_prefixes", []),
            "current": current_world_source_prefixes,
        },
        "current_retained_frames_fallback": current_replay_fallback,
        "evaluations": {
            "reference": [
                _t2_dto_diagnostic("reference", index, dto, ap)
                for index, dto in enumerate(reference_evaluations[:8])],
            "current": [],
        },
    }
    _write_json(dto_diagnostic_path, dto_diagnostics)
    current_evaluations = []
    for index, boundary in enumerate(boundary_records):
        eval_boundary = copy.deepcopy(boundary)
        completion = _complete_causal_generator_metadata(eval_boundary, ap)
        dto_diagnostics["causal_metadata_completion"]["current"].append(
            completion)
        eval_boundary["causal_prestate"]["work_ledger"][
            "_prebuilt_ledger_world_source"] = current_world_sources[index]
        dto = ap.evaluate_boundary(
            eval_boundary, _proof_prediction_config(),
            library_path=current_library, fixture_state=state)
        current_evaluations.append(dto)
        dto_diagnostics["evaluations"]["current"].append(
            _t2_dto_diagnostic("current", index, dto, ap))
        _write_json(dto_diagnostic_path, dto_diagnostics)
        _require_t2_candidate_dto(
            "current", index, dto, dto_diagnostic_path)
        face = dto["candidates"] + dto["inapplicable_candidates"]
        action_names = {row["action"] for row in face}
        if not FOUR_ACTIONS.issubset(action_names):
            raise AssertionError(
                f"T2 boundary did not compare all four actions: {action_names}")
        actions_by_instance: dict[int, set[str]] = {}
        for row in face:
            actions_by_instance.setdefault(
                int(row["instance_id"]), set()).add(str(row["action"]))
        if not actions_by_instance or any(
                not FOUR_ACTIONS.issubset(actions)
                for actions in actions_by_instance.values()):
            raise AssertionError(
                "T2 boundary omitted an action at one or more instances")

    for index, dto in enumerate(reference_evaluations[:8]):
        _require_t2_candidate_dto(
            "reference", index, dto, dto_diagnostic_path)
    comparable = min(len(current_evaluations), len(reference_evaluations), 8)
    if comparable < 2:
        raise AssertionError("T2 reference evaluation prefix lacks P and D")
    def evaluation_face(dto):
        result = copy.deepcopy(dto)
        result.pop("library_sha256", None)
        return result
    for index in range(comparable):
        if evaluation_face(current_evaluations[index]) != \
                evaluation_face(reference_evaluations[index]):
            raise AssertionError(f"T2 full candidate evaluation differs at boundary {index}")
    t2_observations = []
    for index in range(comparable):
        dto = current_evaluations[index]
        face = dto["candidates"] + dto["inapplicable_candidates"]
        t2_observations.append({
            "boundary_index": index,
            "stage": dto["stage"],
            "candidate_count": len(face),
            "applicable_count": len(dto["candidates"]),
            "inapplicable_count": len(dto["inapplicable_candidates"]),
            "approx_costs_ns": {
                str(row["candidate_id"]): row["result"].get("cost_ns")
                for row in dto["approx_results"]
            },
            "exact_costs_ns": {
                str(row["candidate_id"]): row["result"].get("cost_ns")
                for row in dto["exact_results"]
            },
        })

    _verify_bad_input_atomicity(delivery_frames)
    # Prefix equality intentionally stops before administrative request_close
    # because the frozen v2 receiver has no cancellation representation.  The
    # current V4 reader must still accept and journal every actually published
    # frame, including closure when the producer emitted it.
    full_harness = ap._build_dual_ledger_harness(current_protocol)
    for frame in delivery_frames:
        full_harness._apply_work_ledger({"work_ledger": frame})
    full_expected = _ledger_snapshot(
        full_harness, frame_index=len(delivery_frames) - 1)
    _verify_journal_new_process(delivery_frames, full_expected)
    result = {
        "case_id": MICRO_CASE_ID,
        "raw_v2": str(raw_v2_path),
        "raw_v4": str(raw_v4_path),
        "archive_v3": archive_v3_summary,
        "v2_frame_count": len(old_frames),
        "v4_frame_count": len(new_frames),
        "v4_delivery_frame_count": len(delivery_frames),
        "compared_preclose_prefixes": len(current_rows),
        "prefix_state_normalizations": [
            "v2 cursor rows receive cancelled_endpoint_count=0 only on the compared pre-close prefix",
            "v2 descriptor_state declaration and identity copies are checked against their typed pipeline.descriptors entry, then omitted from descriptor_state comparison; the full descriptors maps are still compared exactly",
            "v2 omitted request_closed is represented as false only when the paired V4 pre-close descriptor_state explicitly has false",
            "obligation rows are ordered by obligation_id for comparison; row keys and all row values remain exact",
        ],
        "t2_boundaries_compared": comparable,
        "t2_max_boundaries": T2_MAX_BOUNDARIES,
        "t2_dto_diagnostics": str(dto_diagnostic_path),
        "t2_world_source_prefixes": {
            "reference": legacy.get("world_source_prefixes", []),
            "current": current_world_source_prefixes,
        },
        "t2_retained_frames_fallback": current_replay_fallback,
        "t2_observations": t2_observations,
        "v2_reference_failure_after_prefix": legacy.get("failure"),
        "closure_gap_note": (
            "R_current v2 terminal-tail receiver limitation remains outside "
            "the paired pre-close prefix comparison; V4 cancellation is not "
            "reclassified as physical completion."),
        "bad_input_atomicity": "PASS",
        "physical_ack_evidence": ack_evidence,
        "fresh_process_journal_roundtrip": "PASS",
        "journal_test_scope": (
            "actual BridgeServer writer and fresh-process sidecar reader; "
            "test-only no-op graph response; component evidence only"),
    }
    _write_json(output_path, result)
    return result


def build_d20_construction_case(*, fixture_dir: Path,
                                native_library: Path | None = None) -> dict:
    """Construct a synthetic replica of the historical D geometry only.

    The current scheduler reaches a declared P-completed state and emits one
    eight-iteration D credit batch. Its context and tile geometry match the
    bounded historical descriptor artifact (eight 43-tile groups, 344 tiles
    per descriptor). The 4-rank, 16-layer tiny model keeps this a synthetic
    construction probe, not a full historical D20 model match.
    """
    # Keep all data construction in the ordinary production helper.  It builds
    # no simulator and this command writes only the emitted graph batches.
    drive = _build_production_batches(
        native_library=native_library,
        context_tokens=HISTORICAL_D20_CONTEXT_TOKENS,
        input_tokens=INPUT_TOKENS,
        decode_tokens=HISTORICAL_D20_ITERATIONS,
        train_max_iter=HISTORICAL_D20_ITERATIONS,
        construction_only=True)
    out = fixture_dir / "d20-construction-only"
    out.mkdir(parents=True, exist_ok=True)
    d_batches = [batch for batch in drive["decode_batches"]
                 if any(descriptor.get("stream_kind") ==
                        "decode_remote_credit"
                        for descriptor in batch.get("pipeline_descriptors", []))]
    if len(d_batches) != 1:
        raise AssertionError(
            "D20-shape construction must emit one nonempty eight-step D batch; "
            f"got {len(d_batches)}")
    batches = d_batches
    shape_rows = _descriptor_shape_rows(batches)
    _assert_group_tile_count(
        shape_rows, stream_kind="decode_remote_credit", expected=43,
        label="synthetic four-rank D20-shape D graph")
    if drive["decode_train_iterations"] != [HISTORICAL_D20_ITERATIONS]:
        raise AssertionError(
            "D20-shape construction must emit one 8-iteration decode train; "
            f"got {drive['decode_train_iterations']}")
    for descriptor_row in shape_rows:
        groups = descriptor_row["groups"]
        pass_ordinals = {int(group["pass_or_window_ordinal"])
                         for group in groups}
        axes = [group["tile_axis"] for group in groups]
        if (len(groups) != HISTORICAL_D20_ITERATIONS
                or pass_ordinals != set(range(1, HISTORICAL_D20_ITERATIONS + 1))
                or sum(int(axis["tile_count"]) for axis in axes)
                != HISTORICAL_D20_TOTAL_TILES
                or any(axis["unit"] != "source_tokens"
                       or int(axis["total_units"])
                       != HISTORICAL_D20_CONTEXT_TOKENS
                       or int(axis["units_per_tile"]) != TILE_TOKENS
                       or int(axis["tile_count"]) != 43
                       or int(axis["tail_units"]) != 104
                       for axis in axes)):
            raise AssertionError(
                "D20-shape descriptor must match 8×43 source-token tile "
                f"groups with a 104-token tail: {descriptor_row}")
    slot_reuse = _assert_decode_tile_three_reuse(
        d_batches, label="synthetic four-rank D20-shape D graph")
    runtime = drive["runtime"]
    if (runtime.prefill_complete_ns is None
            or int(runtime.remaining_chunks) != 0):
        raise AssertionError(
            "D20-shape batches require a declared P-completed scheduler state")
    hardware = drive["hardware"]
    expected_hbm = 64 * (1 << 30)
    if int(hardware.local_hbm_capacity_bytes) != expected_hbm:
        raise AssertionError(
            "D20-shape test profile must retain the configured 64 GiB HBM "
            f"capacity, got {hardware.local_hbm_capacity_bytes}")
    model = drive["scheduler"].model
    _write_json(out / "graph_batches.json", [
        _batch_for_tape(batch) for batch in batches])
    report = {
        "case_id": "D20-shape-synthetic-four-rank-344total-8iter",
        "shape_kind": "synthetic-four-rank-current-test-harness",
        "historical_D20_model_match": False,
        "physical_event_loop_started": False,
        "prefill_setup": {
            "scheduler_prefill_complete": True,
            "physical_prefill_simulation_claimed": False,
            "p_graph_constructed_for_lifecycle_setup": True,
            "p_graph_included_as_d20_evidence": False,
            "context_tokens": HISTORICAL_D20_CONTEXT_TOKENS,
            "input_tokens": INPUT_TOKENS,
            "prefill_chunk_size": 512,
        },
        "context_tokens": HISTORICAL_D20_CONTEXT_TOKENS,
        "tile_tokens": TILE_TOKENS,
        "tile_mode": TILE_BYTES,
        "train_max_iter": HISTORICAL_D20_ITERATIONS,
        "tile_count_per_pass": 43,
        "tile_count_per_descriptor": HISTORICAL_D20_TOTAL_TILES,
        "decode_train_iterations": drive["decode_train_iterations"],
        "mesh": [int(hardware.mesh_rows), int(hardware.mesh_cols)],
        "instance_rank_domains": [
            [int(rank) for rank in instance.ranks]
            for instance in drive["topology"].instances],
        "model_shape": {
            key: int(getattr(model, key))
            for key in ("layers", "hidden_size", "ffn_size", "num_heads",
                        "vocab_size", "bytes_per_elem")},
        "per_rank_hbm_capacity_bytes": int(
            hardware.local_hbm_capacity_bytes),
        "batch_count": len(batches),
        "batch_nodes": [len(batch["nodes"]) for batch in batches],
        "descriptor_count": [len(batch["pipeline_descriptors"])
                             for batch in batches],
        "descriptor_shapes": shape_rows,
        "selection_mode": "fixed-remote-read-without-candidate-pricing",
        "selection_observations": drive["observations"],
        "decode_slot_reuse": slot_reuse,
        "descriptor_endpoint_run_count": [
            row["endpoint_run_count"] for row in shape_rows],
        "construction_only_note": (
            "Python production GraphBatchBuilder D output from a declared "
            "P-completed scheduler state with a 64 GiB test capacity profile; "
            "the P graph is setup only. C++ producer, physics, latency and "
            "event-loop runtime were not measured. Synthetic tiny model, not "
            "historical D20 model dimensions."),
    }
    _write_json(out / "construction-report.json", report)
    return report


def validate_historical_descriptors(path: Path) -> dict:
    """Type-parse and check the bounded immutable historical D20 descriptors.

    This reads only the caller-supplied compact extraction artifact. The
    production parser validates all descriptor geometry, affine ID ranges,
    tag runs, anchors, endpoint dependencies, and consume coverage without
    expanding any tile or endpoint.
    """
    from joint.pipeline_descriptor import PipelineDescriptor

    artifact = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(artifact, dict) or not isinstance(
            artifact.get("descriptors"), list):
        raise ValueError("historical descriptor artifact has no descriptor list")
    wrappers = artifact["descriptors"]
    if len(wrappers) != 6:
        raise AssertionError(
            f"historical D20 extraction expected six descriptors, got {len(wrappers)}")

    expected_global_first = list(range(1, HISTORICAL_D20_TOTAL_TILES + 1, 43))
    rows = []
    ranks = set()
    for index, wrapper in enumerate(wrappers):
        declaration = wrapper["declaration"]
        shape = wrapper["shape"]
        wire = declaration["descriptor"]
        descriptor = PipelineDescriptor.from_wire(wire)
        if descriptor.to_wire() != wire:
            raise AssertionError(
                f"historical descriptor {index} failed typed wire roundtrip")
        identity = wire["identity"]
        rank = int(identity["rank"])
        if rank in ranks:
            raise AssertionError(f"duplicate historical descriptor rank {rank}")
        ranks.add(rank)
        if (descriptor.stream_kind != "decode_remote_credit"
                or descriptor.identity.stage != "decode"):
            raise AssertionError(
                f"historical descriptor {index} is not a decode credit stream")
        if declaration.get("identity") != identity:
            raise AssertionError(
                f"historical descriptor {index} declaration identity drift")
        if (shape.get("segment") != descriptor.identity.segment_id
                or shape.get("stream_kind") != descriptor.stream_kind):
            raise AssertionError(
                f"historical descriptor {index} extraction summary identity "
                "drift")

        groups = descriptor.groups
        if len(groups) != HISTORICAL_D20_ITERATIONS:
            raise AssertionError(
                f"historical descriptor {index} has {len(groups)} groups, "
                f"expected {HISTORICAL_D20_ITERATIONS}")
        group_rows = []
        total_tiles = 0
        for group_index, group in enumerate(groups):
            axis = group.tile_axis
            summary = shape["groups"][group_index]
            expected = {
                "ordinal": group.group_ordinal,
                "global_tile_first": expected_global_first[group_index],
                "tail_units": 104,
                "tile_count": 43,
                "total_units": HISTORICAL_D20_CONTEXT_TOKENS,
                "unit": "source_tokens",
                "units_per_tile": TILE_TOKENS,
            }
            actual = {
                "ordinal": int(group.group_ordinal),
                "global_tile_first": int(axis.global_tile_first),
                "tail_units": int(axis.tail_units),
                "tile_count": int(axis.tile_count),
                "total_units": int(axis.total_units),
                "unit": axis.unit,
                "units_per_tile": int(axis.units_per_tile),
            }
            if actual != expected or summary != {
                    "ordinal": expected["ordinal"],
                    "axis": {key: expected[key] for key in (
                        "global_tile_first", "tail_units", "tile_count",
                        "total_units", "unit", "units_per_tile")}}:
                raise AssertionError(
                    f"historical descriptor {index} group {group_index} "
                    f"geometry drift: typed={actual}, summary={summary}")
            total_tiles += int(axis.tile_count)
            group_rows.append(actual)
        endpoint_run_count = len(descriptor.endpoint_runs)
        if (total_tiles != HISTORICAL_D20_TOTAL_TILES
                or endpoint_run_count != 85
                or int(shape["endpoint_runs"])
                != endpoint_run_count):
            raise AssertionError(
                f"historical descriptor {index} has tile/run shape "
                f"{total_tiles}/{endpoint_run_count}, expected 344/85")
        rows.append({
            "rank": rank,
            "request_id": descriptor.identity.request_id,
            "segment_id": descriptor.identity.segment_id,
            "buffer_domain_id": descriptor.identity.buffer_domain_id,
            "stream_kind": descriptor.stream_kind,
            "group_count": len(groups),
            "group_axes": group_rows,
            "tile_count_per_descriptor": total_tiles,
            "endpoint_run_count": endpoint_run_count,
            "endpoint_dependency_run_count": len(
                descriptor.endpoint_dependency_runs),
            "consume_completion_run_count": len(
                descriptor.consume_completion_runs),
            "logical_id_range_count": len(descriptor.logical_id_ranges),
            "tag_allocation_run_count": len(descriptor.tag_allocation_runs),
            "anchor_count": len(descriptor.anchors),
        })
    if sorted(ranks) != [0, 1, 6, 7, 12, 13]:
        raise AssertionError(
            f"historical D20 descriptor ranks changed: {sorted(ranks)}")
    return {
        "case_id": "historical-D20-bounded-descriptor-validation",
        "artifact_path": str(path.resolve()),
        "artifact_sha256": _sha256_file(path),
        "bounded_read": artifact.get("bounded_read"),
        "descriptor_count": len(rows),
        "context_tokens": HISTORICAL_D20_CONTEXT_TOKENS,
        "tile_tokens": TILE_TOKENS,
        "groups_per_descriptor": HISTORICAL_D20_ITERATIONS,
        "tiles_per_group": 43,
        "tiles_per_descriptor": HISTORICAL_D20_TOTAL_TILES,
        "tail_tokens": 104,
        "validation": (
            "PipelineDescriptor.from_wire PASS: geometry, affine IDs, tags, "
            "anchors, endpoint dependencies, and consume coverage"),
        "descriptors": rows,
    }


def _arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-dir", type=Path)
    parser.add_argument("--native-library", type=Path)
    parser.add_argument("--generate-only", action="store_true",
                        help="generate the micro input, without launching C++")
    parser.add_argument("--d20-construction-only", action="store_true",
                        help="separately build D20 graph input without C++ event loop")
    parser.add_argument("--raw-v2", type=Path,
                        help="R_current frozen-binary raw result from this exact tape")
    parser.add_argument("--raw-v4", type=Path,
                        help="current V4-binary raw result from this exact tape")
    parser.add_argument("--archive-v3", type=Path,
                        help="optional retained V3 raw archive; offline replay only")
    parser.add_argument("--reference-root", type=Path,
                        help="frozen R_current source root")
    parser.add_argument("--reference-library", type=Path,
                        help="same-ABI projection library from frozen R_current")
    parser.add_argument("--current-library", type=Path,
        help="same-ABI projection library for current V4")
    parser.add_argument("--comparison-output", type=Path)
    parser.add_argument(
        "--historical-descriptors", type=Path,
        help="pure typed validation of a caller-supplied bounded D20 extraction")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _arg_parser().parse_args(argv)
    if args.historical_descriptors is not None:
        result = validate_historical_descriptors(
            args.historical_descriptors.resolve())
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.generate_only or args.d20_construction_only:
        if args.fixture_dir is None:
            raise SystemExit(
                "--fixture-dir is required for fixture generation")
        if args.d20_construction_only:
            report = build_d20_construction_case(
                fixture_dir=args.fixture_dir)
            print(json.dumps(report, indent=2, sort_keys=True))
            return 0
        if args.native_library is None:
            raise SystemExit(
                "--native-library is required for full-face micro construction")
        provenance = _write_micro_case(
            args.fixture_dir.resolve(), args.native_library.resolve())
        print(json.dumps(provenance, indent=2, sort_keys=True))
        return 0
    required = (args.raw_v2, args.raw_v4, args.reference_root,
                args.reference_library, args.current_library)
    if any(value is None for value in required):
        raise SystemExit(
            "comparison requires --raw-v2 --raw-v4 --reference-root "
            "--reference-library and --current-library")
    if args.fixture_dir is None:
        raise SystemExit("--fixture-dir is required for comparison")
    output_path = (args.comparison_output.resolve()
                   if args.comparison_output else
                   args.fixture_dir.resolve() / "sparse-pipeline-comparison.json")
    result = compare_raw_pair(
        fixture_dir=args.fixture_dir.resolve(),
        raw_v2_path=args.raw_v2.resolve(),
        raw_v4_path=args.raw_v4.resolve(),
        reference_root=args.reference_root.resolve(),
        current_library=args.current_library.resolve(),
        reference_library=args.reference_library.resolve(),
        output_path=output_path,
        archive_v3_path=(None if args.archive_v3 is None
                         else args.archive_v3.resolve()))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
