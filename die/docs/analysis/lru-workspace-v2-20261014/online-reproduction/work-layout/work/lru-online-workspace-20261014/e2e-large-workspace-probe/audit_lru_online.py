#!/usr/bin/env python3
"""Cross-check captured LRU Online wire, workspace, and full-detail metrics."""

from __future__ import annotations

import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path


ROOT = next(p for p in Path(__file__).resolve().parents if (p / "die/src/Execution.cc").is_file())
E2E = ROOT / "work/lru-online-workspace-20261014/e2e-large-workspace-probe"
RUN = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else E2E / "run"
BASE_RUN = ROOT / "work/online-baseline-20261014/e2e/run"
LRU_LEGACY_RUN = ROOT / "work/lru-online-workspace-20261014/e2e/run"
OUT = RUN / "wire_audit/lru_online_validation.json"
errors: list[str] = []
checks: dict[str, object] = {}


def require(condition: bool, label: str) -> None:
    if not condition:
        errors.append(label)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def records_from_metrics(path: Path) -> dict[str, list[dict]]:
    result: dict[str, list[dict]] = defaultdict(list)
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        marker = "[METRIC] "
        if marker not in line:
            continue
        try:
            record = json.loads(line.split(marker, 1)[1])
        except Exception as exc:  # noqa: BLE001
            errors.append(f"invalid full-detail metric JSON: {exc}")
            continue
        kind = record.get("type")
        if isinstance(kind, str):
            result[kind].append(record)
    return result


def json_type(value: object) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "number"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


def flatten_types(value: object, prefix: str = "$") -> dict[str, set[str]]:
    result: dict[str, set[str]] = defaultdict(set)
    result[prefix].add(json_type(value))
    if isinstance(value, dict):
        for key, child in value.items():
            for path, types in flatten_types(child, f"{prefix}.{key}").items():
                result[path].update(types)
    elif isinstance(value, list):
        for child in value:
            for path, types in flatten_types(child, f"{prefix}[]").items():
                result[path].update(types)
    return result


def flatten_values(value: object, prefix: str = "$") -> dict[str, list[object]]:
    result: dict[str, list[object]] = defaultdict(list)
    if isinstance(value, dict):
        for key, child in value.items():
            for path, values in flatten_values(child, f"{prefix}.{key}").items():
                result[path].extend(values)
    elif isinstance(value, list):
        for child in value:
            for path, values in flatten_values(child, f"{prefix}[]").items():
                result[path].extend(values)
    else:
        result[prefix].append(value)
    return result


def metric_schema_audit(actual: dict[str, list[dict]],
                        lru_baseline: dict[str, list[dict]],
                        base_reference: dict[str, list[dict]]) -> dict:
    # Preserve the actual pre-port LRU emission contract. The generic base
    # run is diagnostic only because some base-only metrics have no LRU meaning.
    audited = (
        "rank_compute", "local_hbm", "iteration", "request", "summary",
        "consistency", "memory_anchor", "hbm_watermark_summary", "init",
        "capacity_timeavg",
    )
    result: dict[str, dict] = {}
    for kind in audited:
        rows = actual.get(kind, [])
        legacy_rows = lru_baseline.get(kind, [])
        reference_rows = base_reference.get(kind, [])
        if not rows:
            result[kind] = {
                "observed_count": 0,
                "baseline_count": len(legacy_rows),
                "coverage": "not_observed_in_tiny_run",
            }
            continue
        if not legacy_rows:
            result[kind] = {
                "observed_count": len(rows),
                "baseline_count": 0,
                "additional_observed_keys": sorted(
                    set().union(*(set(row) for row in rows))),
                "coverage": "observed_no_pre_port_LRU_schema_reference",
            }
            continue
        expected_keys = set().union(*(set(row) for row in legacy_rows))
        present_keys = set().union(*(set(row) for row in rows))
        missing_keys = sorted(expected_keys - present_keys)
        additional_keys = sorted(present_keys - expected_keys)
        if missing_keys:
            errors.append(f"metric {kind}: missing pre-port LRU keys {missing_keys}")
        type_errors: list[str] = []
        for key in sorted(expected_keys & present_keys):
            accepted = {json_type(row[key]) for row in legacy_rows
                        if key in row and row[key] is not None}
            observed = {json_type(row[key]) for row in rows
                        if key in row and row[key] is not None}
            if accepted and not observed.issubset(accepted):
                type_errors.append(f"{key}: expected {sorted(accepted)}, got {sorted(observed)}")
        if type_errors:
            errors.append(f"metric {kind}: pre-port LRU value type changes: {type_errors}")
        legacy_paths: dict[str, set[str]] = defaultdict(set)
        actual_paths: dict[str, set[str]] = defaultdict(set)
        for row in legacy_rows:
            for path, types in flatten_types(row).items():
                legacy_paths[path].update(types)
        for row in rows:
            for path, types in flatten_types(row).items():
                actual_paths[path].update(types)
        missing_paths = sorted(set(legacy_paths) - set(actual_paths))
        additional_paths = sorted(set(actual_paths) - set(legacy_paths))
        nested_type_errors = []
        for path in sorted(set(legacy_paths) & set(actual_paths)):
            accepted = legacy_paths[path] - {"null"}
            observed = actual_paths[path] - {"null"}
            if accepted and not observed.issubset(accepted):
                nested_type_errors.append(
                    f"{path}: expected {sorted(accepted)}, got {sorted(observed)}")
        if missing_paths:
            errors.append(
                f"metric {kind}: missing pre-port LRU nested fields {missing_paths}")
        if nested_type_errors:
            errors.append(f"metric {kind}: pre-port LRU nested type changes {nested_type_errors}")

        base_keys = set().union(*(set(row) for row in reference_rows)) \
            if reference_rows else set()
        base_only_keys = sorted(base_keys - present_keys)
        flat_zero_watermark = (
            kind == "hbm_watermark_summary" and
            all(row.get("sample_count") == 0 and
                row.get("source") == "planner_memory_ledger" and
                isinstance(row.get("note"), str) and
                "flat-zero watermark" in row.get("note", "")
                for row in rows)
        )
        unmapped_base_only = (
            sorted(set(base_only_keys) & {"window_start_ns", "window_end_ns"})
            if flat_zero_watermark else []
        )
        result[kind] = {
            "observed_count": len(rows),
            "baseline_count": len(legacy_rows),
            "legacy_key_count": len(expected_keys),
            "missing_legacy_keys": missing_keys,
            "additional_observed_keys": additional_keys,
            "value_type_errors": type_errors,
            "legacy_nested_field_count": len(legacy_paths),
            "missing_nested_legacy_fields": missing_paths,
            "additional_observed_nested_fields": additional_paths,
            "nested_value_type_errors": nested_type_errors,
            "base_variant_only_fields": base_only_keys,
            "unmapped_base_variant_only_fields": unmapped_base_only,
            "base_variant_unmapped_reason": (
                "pre-port LRU flat-zero watermark has no planner replay/window bounds"
                if unmapped_base_only else None
            ),
            "coverage": "checked_against_pre_port_LRU_full_detail_schema",
        }
    return result


require(RUN.is_dir(), f"missing run directory: {RUN}")
if not errors:
    wire_dir = RUN / "wire_audit"
    cpp_log = (RUN / "cpp.log").read_text(encoding="utf-8", errors="replace")
    python_log = (RUN / "python.log").read_text(encoding="utf-8", errors="replace")
    lru_legacy_log_path = LRU_LEGACY_RUN / "cpp.log"
    lru_legacy_metrics = records_from_metrics(lru_legacy_log_path)
    actual_metrics = records_from_metrics(RUN / "cpp.log")
    base_metrics = records_from_metrics(BASE_RUN / "cpp.log")
    requests = [load_json(p) for p in sorted(wire_dir.glob("request_*.json"),
                                              key=lambda p: int(p.stem.split("_")[-1]))]
    responses = [load_json(p) for p in sorted(wire_dir.glob("response_*.json"),
                                               key=lambda p: int(p.stem.split("_")[-1]))]
    acks = [load_json(p) for p in sorted(wire_dir.glob("ack_*.json"),
                                           key=lambda p: int(p.stem.split("_")[-1]))]
    compiled = load_json(wire_dir / "compiled_plans.json")
    wire_summary = load_json(wire_dir / "audit.json")
    exits = {}
    for line in (RUN / "exit_codes.txt").read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            exits[key] = int(value)

    with (E2E / "workspace_e2e_request_queue_recompute.csv").open(
            newline="", encoding="utf-8") as stream:
        queue_rows = list(csv.DictReader(stream))
    require(len(queue_rows) == 1, "tiny official queue must contain exactly one request")
    expected_identity = None
    if len(queue_rows) == 1:
        row = queue_rows[0]
        expected_identity = {
            "session_id": row["session_id"],
            "turn_index": int(row["turn_index"]),
            "request_id": row["request_id"],
        }

    require(exits.get("cpp_exit") == 0 and exits.get("python_exit") == 0,
            f"Online process exits are not both zero: {exits}")
    require(bool(wire_summary.get("provenance_gate_matched")),
            "official request queue provenance gate did not match")
    require(len(requests) == len(responses) == len(acks) == 3,
            f"expected three actual wire exchanges, saw req/resp/ack {len(requests)}/{len(responses)}/{len(acks)}")
    require("REQUEST_COMPLETE" in requests[-1].get("reasons", []) if requests else False,
            "no terminal REQUEST_COMPLETE was captured")
    require(bool(wire_summary.get("actual_request_complete_observed")),
            "wire audit did not recognize REQUEST_COMPLETE")
    require(wire_summary.get("protocol_schema_valid") is True and
            not wire_summary.get("protocol_errors"),
            f"base schema/workspace wire audit failed: {wire_summary.get('protocol_errors')}")

    hardware = load_json(E2E / "hardware_coarse_8rank.json")
    mesh = hardware.get("mesh", {})
    rank_count = mesh.get("rows", 0) * mesh.get("columns", 0)
    require("provenance sidecar: matched" in cpp_log,
            "C++ run did not report the official request queue provenance gate")
    require(mesh.get("rows") == 4 and mesh.get("columns") == 2 and
            rank_count == 8,
            "official fixture hardware is not the intended full 4x2 topology")
    # The official trace config must preserve the full P/D rank assignments.
    with (E2E / "trace_config.csv").open(newline="", encoding="utf-8") as stream:
        config_rows = list(csv.DictReader(stream))
    groups = [row for row in config_rows if row.get("kind") == "inference_group"]
    group_summary = sorted((g["phase_role"], g["ranks"], g["pg_name"]) for g in groups)
    require(group_summary == [
                ("decode", "4,5,6,7", "2"),
                ("prefill", "0,1,2,3", "1"),
            ],
            f"official 2x2 P/D rank assignment changed: {group_summary}")
    config_values = {row["key"]: row["value"] for row in config_rows
                     if row.get("kind") == "config"}
    require(config_values.get("layers") == "1" and
            config_values.get("hidden_size") == "512" and
            config_values.get("ffn_size") == "1024" and
            config_values.get("vocab_size") == "512" and
            config_values.get("num_heads") == "4" and
            config_values.get("mlp_variant") == "swiglu",
            f"tiny model dimensions changed: {config_values}")
    require(config_values.get("kv_cache_policy") == "session_lru_tiered" and
            config_values.get("model_name") ==
            "workspace_e2e_probe_mha4_h512_f1024_gqa1",
            "fixture is not using the official LRU tiered policy / declared MHA model")
    plan_dirs = sorted(E2E.joinpath("generated").glob(
        "llama2_7b_wsc_llm_inference_54npus_plan_*"))
    require(len(plan_dirs) == 1,
            f"expected one official-format materialized plan directory, got {plan_dirs}")
    if len(plan_dirs) == 1:
        manifest = load_json(plan_dirs[0] / "manifest.json")
        require(manifest.get("hardware", {}).get("capacity_profile") ==
                "workspace-e2e-coarse" and
                manifest.get("kv_management", {}).get("policy") ==
                "session_lru_tiered" and
                manifest.get("model", {}).get("name") ==
                "workspace_e2e_probe_mha4_h512_f1024_gqa1",
                "official plan manifest lost fixture model, HBM profile, or LRU policy")

    # Schema-2 request/response/ack round trips retain sequence binding.
    for sequence, (request, response, ack) in enumerate(zip(requests, responses, acks)):
        require(request.get("schema_version") == 2 and
                request.get("delivery_sequence") == sequence,
                f"request {sequence}: schema2 sequence binding failed")
        require(response.get("schema_version") == 2 and
                response.get("source_delivery_sequence") == sequence and
                response.get("batch_id") == sequence,
                f"response {sequence}: schema2 sequence binding failed")
        require(ack.get("schema_version") == 2 and
                ack.get("delivery_sequence") == sequence and
                ack.get("batch_id") == sequence and ack.get("success") is True,
                f"ack {sequence}: schema2 sequence binding or success failed")

    if expected_identity and requests:
        arrival_rows = [a for req in requests for a in req.get("arrivals", [])]
        arrival_identities = [{key: a.get(key) for key in expected_identity}
                              for a in arrival_rows]
        require(arrival_identities == [expected_identity],
                f"legacy queue identity/request key not preserved in arrivals: {arrival_identities}")
        decision_path = RUN / "bridge/online_decision_log.jsonl"
        decisions = [json.loads(line) for line in decision_path.read_text(
            encoding="utf-8").splitlines() if line.strip()]
        require(bool(decisions) and all(d.get("request_id") == expected_identity["request_id"]
                                        for d in decisions),
                "scheduler decision journal lost the legacy request_id key")
        metric_requests = actual_metrics.get("request", [])
        require(len(metric_requests) == 1 and all(
            metric_requests[0].get(key) == value for key, value in expected_identity.items()),
            "full-detail request record lost session_id/request_id/turn_index legacy keys")
        kv_journal_path = RUN / "bridge/kv_delta_journal.jsonl"
        kv_rows = [json.loads(line) for line in kv_journal_path.read_text(
            encoding="utf-8").splitlines() if line.strip()]
        require(any(r.get("session_id") == expected_identity["session_id"]
                    for r in kv_rows),
                "KV delta journal did not preserve the legacy session_id key")
        checks["legacy_request_identity"] = {
            "expected": expected_identity,
            "arrival_identities": arrival_identities,
            "scheduler_decision_rows": len(decisions),
            "request_metric_rows": len(metric_requests),
            "kv_delta_rows": len(kv_rows),
            "kv_rows_with_session_key": sum(
                r.get("session_id") == expected_identity["session_id"] for r in kv_rows),
        }

    # Every response carries the complete rank-local capacity view and the
    # sum of non-workspace commitments plus active transient workspace fits.
    states_by_sequence: dict[int, dict[int, dict]] = {}
    for sequence, request in enumerate(requests):
        state = request.get("workspace_state", {})
        rows = state.get("ranks", []) if isinstance(state, dict) else []
        require(state.get("protocol_version") == 1 and len(rows) == rank_count and
                [r.get("rank") for r in rows] == list(range(rank_count)),
                f"request {sequence}: workspace state must cover all ranks")
        states_by_sequence[sequence] = {r.get("rank"): r for r in rows}
    live_workspace_samples = [
        {"delivery_sequence": sequence, "rank": rank,
         "active_workspace_bytes": row.get("active_workspace_bytes")}
        for sequence, rank_rows in sorted(states_by_sequence.items())
        for rank, row in sorted(rank_rows.items())
        if isinstance(row.get("active_workspace_bytes"), int) and
        not isinstance(row.get("active_workspace_bytes"), bool) and
        row.get("active_workspace_bytes", 0) > 0
    ]
    capacity_rows = []
    for sequence, response in enumerate(responses):
        snapshot = response.get("workspace_capacity_snapshot", {})
        rows = snapshot.get("ranks", []) if isinstance(snapshot, dict) else []
        require(snapshot.get("protocol_version") == 1 and len(rows) == rank_count and
                [r.get("rank") for r in rows] == list(range(rank_count)),
                f"response {sequence}: workspace capacity snapshot must cover all ranks")
        for row in rows:
            rank = row.get("rank")
            state_row = states_by_sequence.get(sequence, {}).get(rank, {})
            revision = row.get("workspace_revision")
            require(revision == state_row.get("workspace_revision"),
                    f"response {sequence} rank {rank}: capacity revision does not echo request")
            active = state_row.get("active_workspace_bytes", 0)
            base_bytes = (row.get("model_weight_bytes", 0) +
                          row.get("resident_kv_bytes", 0) +
                          row.get("reserved_request_bytes", 0))
            fits = (isinstance(row.get("capacity_bytes"), int) and
                    base_bytes <= row["capacity_bytes"] and
                    isinstance(active, int) and
                    active <= row["capacity_bytes"] - base_bytes)
            require(fits, f"response {sequence} rank {rank}: weights+KV+reserved+workspace exceed capacity")
            capacity_rows.append({
                "delivery_sequence": sequence,
                "rank": rank,
                "workspace_revision": revision,
                "capacity_bytes": row.get("capacity_bytes"),
                "model_weight_bytes": row.get("model_weight_bytes"),
                "resident_kv_bytes": row.get("resident_kv_bytes"),
                "reserved_request_bytes": row.get("reserved_request_bytes"),
                "active_workspace_bytes": active,
                "fits": fits,
            })

    # The revision changes after each acquire and release. Match nonzero
    # compiled workspace Plans to the following bridge request state, require
    # a positive even revision delta, and require all leases to be returned.
    positive_items = [item for item in compiled.get("items", [])
                      if item.get("workspace_positive") is True]
    positive_plan_ranks = sorted({item.get("rank") for item in positive_items})
    # A zero-workspace Plan still exercises the production graph and admission
    # path, but it cannot demonstrate a positive live hold. Report that coverage
    # gap without treating it as a protocol failure; a later positive-workspace
    # fixture must provide positive Plans on every rank in its P/D groups.
    if positive_items:
        require(positive_plan_ranks == list(range(rank_count)),
                f"tiny 4x2 P/D fixture needs positive typed workspace on all ranks, saw {positive_plan_ranks}")
    typed_specs = [
        (node, node.get("compute", {}).get("die_op_spec", {}))
        for response in responses for node in response.get("nodes", [])
        if isinstance(node.get("compute", {}).get("die_op_spec"), dict)
    ]
    model_parallel_rows = []
    for node, spec in typed_specs:
        dims = spec.get("model_dims", {})
        tp = spec.get("tp_slice", {})
        expected_dims = {
            "hidden": 512, "ffn": 1024, "vocab": 512,
            "num_q_heads": 4, "num_kv_heads": 4, "head_dim": 128,
            "mlp_variant": "swiglu",
        }
        require(all(dims.get(key) == value for key, value in expected_dims.items()),
                f"typed model dims do not match declared MHA/GQA1 tiny model: {dims}")
        rank = node.get("rank")
        group_local_rank = (rank % 4 if isinstance(rank, int) else None)
        require(tp.get("tensor_parallel") == 4 and
                tp.get("tensor_parallel_rank") == group_local_rank and
                tp.get("q_heads_local") == 1 and
                tp.get("kv_heads_local") == 1 and
                tp.get("ffn_local") == 256,
                f"rank {node.get('rank')} typed TP slice is not group-local 1Q/1KV/256FFN: {tp}")
        model_parallel_rows.append({
            "rank": node.get("rank"),
            "op_kind": spec.get("op_kind"),
            "num_q_heads": dims.get("num_q_heads"),
            "num_kv_heads": dims.get("num_kv_heads"),
            "head_dim": dims.get("head_dim"),
            "tensor_parallel": tp.get("tensor_parallel"),
            "q_heads_local": tp.get("q_heads_local"),
            "kv_heads_local": tp.get("kv_heads_local"),
            "ffn_local": tp.get("ffn_local"),
        })
    require(bool(typed_specs), "official responses carried no typed model specs")
    checks["model_parallel_contract"] = {
        "model_identity": "MHA/GQA1: Q=4 KV=4 head_dim=128",
        "tensor_parallel": 4,
        "typed_spec_count": len(typed_specs),
        "spec_slices": model_parallel_rows,
    }
    lease_evidence = []
    acquisition_counts = {rank: 0 for rank in range(rank_count)}
    release_counts = {rank: 0 for rank in range(rank_count)}
    positive_items_by_stage_rank: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for item in positive_items:
        sequence = item.get("delivery_sequence")
        rank = item.get("rank")
        if isinstance(sequence, int) and isinstance(rank, int):
            positive_items_by_stage_rank[(sequence, rank)].append(item)
    for (sequence, rank), plans in sorted(positive_items_by_stage_rank.items()):
        before = states_by_sequence.get(sequence, {}).get(rank)
        after = states_by_sequence.get(sequence + 1, {}).get(rank)
        require(before is not None and after is not None,
                f"positive workspace plan at response {sequence}, rank {rank} has no post-execution state")
        if before is None or after is None:
            continue
        delta = after.get("workspace_revision", 0) - before.get("workspace_revision", 0)
        require(delta > 0 and delta % 2 == 0,
                f"rank {rank}: workspace revision delta {delta} does not show balanced acquire/release")
        require(after.get("active_workspace_bytes") == 0,
                f"rank {rank}: positive workspace remains active after the completed stage")
        if delta > 0 and delta % 2 == 0:
            acquisition_counts[rank] += delta // 2
            release_counts[rank] += delta // 2
        lease_evidence.append({
            "delivery_sequence": sequence,
            "rank": rank,
            "positive_plan_count": len(plans),
            "op_kinds": sorted({item.get("op_kind") for item in plans}),
            "hbm_workspace_peak_bytes": sorted({
                item.get("hbm_workspace_peak_bytes") for item in plans
            }),
            "revision_before": before.get("workspace_revision"),
            "revision_after": after.get("workspace_revision"),
            "inferred_acquires": delta // 2 if delta > 0 and delta % 2 == 0 else None,
            "active_after": after.get("active_workspace_bytes"),
        })
    for rank in range(rank_count):
        state_rows = [states_by_sequence.get(seq, {}).get(rank, {})
                      for seq in sorted(states_by_sequence)]
        revisions = [row.get("workspace_revision", -1) for row in state_rows]
        require(all(a <= b for a, b in zip(revisions, revisions[1:])),
                f"rank {rank}: workspace revisions are not monotonic: {revisions}")
        if state_rows:
            require(state_rows[-1].get("active_workspace_bytes") == 0,
                    f"rank {rank}: final active workspace is not zero")
        if positive_items:
            require(acquisition_counts[rank] == release_counts[rank] and
                    acquisition_counts[rank] > 0,
                    f"rank {rank}: positive-workspace acquire/release evidence is not balanced/nonzero")
    checks["workspace_leases"] = {
        "positive_plan_ranks": positive_plan_ranks,
        "inferred_acquisitions_by_rank": acquisition_counts,
        "inferred_releases_by_rank": release_counts,
        "positive_live_workspace_observed": bool(live_workspace_samples),
        "positive_live_workspace_samples": live_workspace_samples,
        "positive_workspace_plan_coverage": (
            "positive_plans_observed" if positive_items else
            "not_observed_all_compiled_plan_peaks_are_zero"),
        "observation_scope": (
            "bridge-boundary workspace_state snapshots only; no phase-wide live sample"
            if not live_workspace_samples else
            "positive active bytes observed in a bridge-boundary workspace_state snapshot"),
        "plans": lease_evidence,
        "capacity_checks": capacity_rows,
    }

    # Markers are protocol/control nodes and must never enter the typed math
    # path. The full-detail Die category must carry no INT4/vector math.
    marker_nodes = [node for response in responses for node in response.get("nodes", [])
                    if node.get("compute", {}).get("op_role") == "control_marker"]
    require(bool(marker_nodes), "no control markers observed in the tiny Online graph")
    marker_has_spec = [node.get("name") for node in marker_nodes
                       if "die_op_spec" in node.get("compute", {})]
    require(not marker_has_spec, f"control markers carried typed DIE math specs: {marker_has_spec}")
    rank_rows = {row.get("rank"): row for row in actual_metrics.get("rank_compute", [])}
    local_hbm_rows = {row.get("rank"): row for row in actual_metrics.get("local_hbm", [])}
    math_by_rank = {}
    typed_metric_rows = {"rank_compute": [], "local_hbm": []}
    for rank in range(rank_count):
        metric = rank_rows.get(rank, {})
        coarse_runtime = metric.get("coarse_die_runtime")
        if not isinstance(coarse_runtime, dict):
            require(False, f"rank {rank}: newly ported coarse_die_runtime is missing")
            math_by_rank[rank] = {
                "runtime_math_coverage": "not_observed_coarse_die_runtime_missing",
                "comp_int4_ops": None,
                "comp_vec_eqops": None,
                "marker_int4_ops": None,
                "marker_vec_eqops": None,
            }
            continue
        cancelled = metric.get("die_cancelled_comp_count")
        require(isinstance(cancelled, int) and not isinstance(cancelled, bool) and
                cancelled >= 0,
                f"rank {rank}: die_cancelled_comp_count is missing or not an integer")
        runtime_window_ns = coarse_runtime.get("utilization_window_ns")
        require(coarse_runtime.get("utilization_window") == "[0, sim_end_ns]" and
                isinstance(runtime_window_ns, int) and runtime_window_ns > 0,
                f"rank {rank}: coarse typed utilization window is invalid")
        categories = coarse_runtime.get("served_by_category", {})
        require(isinstance(categories, dict) and
                {"comm_recv", "comm_send", "comp", "die", "pool", "restore"}
                .issubset(categories),
                f"rank {rank}: coarse runtime category map is incomplete")
        comp = categories.get("comp", {})
        marker = categories.get("die", {})
        comp_math = (comp.get("int4_ops", 0), comp.get("vec_eqops", 0))
        marker_math = (marker.get("int4_ops", 0), marker.get("vec_eqops", 0))
        require(comp_math[0] > 0 and comp_math[1] > 0,
                f"rank {rank}: typed COMP math was not observed: {comp_math}")
        require(marker_math == (0, 0),
                f"rank {rank}: marker/DIE category contains typed math: {marker_math}")
        math_by_rank[rank] = {"comp_int4_ops": comp_math[0],
                              "comp_vec_eqops": comp_math[1],
                              "marker_int4_ops": marker_math[0],
                              "marker_vec_eqops": marker_math[1],
                              "runtime_math_coverage": "observed"}
        typed_metric_rows["rank_compute"].append({
            "rank": rank,
            "coarse_die_runtime_window_ns": runtime_window_ns,
            "die_cancelled_comp_count": cancelled,
            "served_categories": sorted(categories),
        })

        local_metric = local_hbm_rows.get(rank, {})
        local_categories = local_metric.get("served_by_category", {})
        local_window_ns = local_metric.get("utilization_window_ns")
        require(local_metric.get("accounting_history_complete") is True and
                local_metric.get("source") == "shared_coarse_die_runtime_model" and
                local_metric.get("utilization_window") == "[0, sim_end_ns]" and
                isinstance(local_window_ns, int) and local_window_ns > 0,
                f"rank {rank}: newly ported local_hbm typed accounting/window fields are incomplete")
        require(isinstance(local_categories, dict) and
                {"comm_recv", "comm_send", "comp", "die", "pool", "restore"}
                .issubset(local_categories),
                f"rank {rank}: local_hbm typed category map is incomplete")
        local_comp = local_categories.get("comp", {})
        local_die = local_categories.get("die", {})
        for label, category in (("comp", local_comp), ("die", local_die)):
            require(isinstance(category, dict) and
                    isinstance(category.get("int4_ops"), (int, float)) and
                    not isinstance(category.get("int4_ops"), bool) and
                    isinstance(category.get("vec_eqops"), (int, float)) and
                    not isinstance(category.get("vec_eqops"), bool) and
                    isinstance(category.get("hbm_served_bytes_by_row_direction"), list) and
                    isinstance(category.get("hbm_served_bytes_by_row_direction_exact"), list),
                    f"rank {rank}: local_hbm {label} typed math/HBM rows are incomplete")
        typed_metric_rows["local_hbm"].append({
            "rank": rank,
            "utilization_window_ns": local_window_ns,
            "accounting_history_complete": local_metric.get("accounting_history_complete"),
            "served_categories": sorted(local_categories)
                if isinstance(local_categories, dict) else [],
        })
    checks["coarse_typed_metrics"] = typed_metric_rows
    checks["typed_math_and_markers"] = {
        "control_marker_count": len(marker_nodes),
        "marker_typed_spec_nodes": marker_has_spec,
        "rank_math": math_by_rank,
    }

    # REQUEST_COMPLETE must preserve the request key and produce the terminal
    # logical group; the C++ lifecycle must drain all terminals/watches.
    complete_groups = [group for request in requests
                       if "REQUEST_COMPLETE" in request.get("reasons", [])
                       for group in request.get("completed_groups", [])
                       if group.get("stage") == ""]
    require(len(complete_groups) == 1 and expected_identity and
            complete_groups[0].get("request_id") == expected_identity["request_id"],
            f"REQUEST_COMPLETE did not close the legacy request key: {complete_groups}")
    lifecycle_match = re.search(
        r"phase-4 end audit:.*terminal_total=(\d+) terminal_success=(\d+) "
        r"terminal_skipped=(\d+) terminal_other=(\d+)", cpp_log)
    require(lifecycle_match is not None, "C++ terminal lifecycle audit line is missing")
    if lifecycle_match:
        total, success, skipped, other = map(int, lifecycle_match.groups())
        require(total == success and skipped == 0 and other == 0,
                f"C++ terminal ledger did not drain cleanly: {lifecycle_match.groups()}")
        checks["terminal_lifecycle"] = {
            "terminal_total": total,
            "terminal_success": success,
            "terminal_skipped": skipped,
            "terminal_other": other,
        }
    service_match = re.search(
        r"service counters: accepted=(\d+) completed=(\d+) active=(\d+) pending_alarm=(\d+)",
        cpp_log)
    require(service_match is not None, "C++ Online service counter line is missing")
    if service_match:
        accepted, completed, active, alarms = map(int, service_match.groups())
        require((accepted, completed, active, alarms) == (1, 1, 0, 0),
                f"C++ request service accounting is not clean: {service_match.groups()}")
        checks["service_counters"] = {
            "accepted": accepted,
            "completed": completed,
            "active": active,
            "pending_alarm": alarms,
        }

    # Keep the old full-detail record contract visible: compare only emitted
    # records, and explicitly report record kinds the tiny trace did not emit.
    schema = metric_schema_audit(actual_metrics, lru_legacy_metrics, base_metrics)
    unit_named_fields: dict[str, dict[str, object]] = {}
    unit_suffixes = ("_ns", "_bytes", ".bytes", "_ops", "_eqops",
                     "_Bps", "_ops_per_s")
    for kind, rows in actual_metrics.items():
        leaves: dict[str, list[object]] = defaultdict(list)
        baseline_leaves: dict[str, list[object]] = defaultdict(list)
        for row in rows:
            for path, values in flatten_values(row).items():
                if path.endswith(unit_suffixes) or "_bytes_per_second" in path:
                    leaves[path].extend(values)
        for row in lru_legacy_metrics.get(kind, []):
            for path, values in flatten_values(row).items():
                if path.endswith(unit_suffixes) or "_bytes_per_second" in path:
                    baseline_leaves[path].extend(values)
        field_errors = []
        field_types: dict[str, dict[str, list[str]]] = {}
        for path, values in sorted(leaves.items()):
            observed = sorted({json_type(value) for value in values if value is not None})
            expected = sorted({json_type(value) for value in baseline_leaves.get(path, [])
                               if value is not None})
            # If the base record never carried this path, there is no baseline
            # type contract to compare. Preserve the observed type and mark it
            # as baseline-unreferenced instead of inventing a numeric rule.
            invalid = ([type_name for type_name in observed
                        if type_name not in expected]
                       if expected else [])
            if invalid:
                field_errors.append({"path": path, "baseline_types": expected,
                                     "observed_types": observed,
                                     "unexpected_types": invalid})
            field_types[path] = {"baseline": expected, "observed": observed}
        if field_errors:
            errors.append(f"metric {kind}: unit-named field type drift {field_errors}")
        unit_named_fields[kind] = {
            "field_count": len(leaves),
            "types_by_field": field_types,
            "invalid_types": field_errors,
        }
    checks["legacy_metric_schema"] = {
        "reference_run": str(LRU_LEGACY_RUN),
        "reference_cpp_log_sha256": __import__("hashlib").sha256(
            lru_legacy_log_path.read_bytes()).hexdigest(),
        "base_variant_comparison_run": str(BASE_RUN),
        "record_counts": {kind: len(rows) for kind, rows in sorted(actual_metrics.items())},
        "records": schema,
        "unit_named_fields": unit_named_fields,
        "unobserved_legacy_record_types": [kind for kind, value in schema.items()
                                           if value["coverage"] == "not_observed_in_tiny_run"],
    }
    required_metric_counts = {
        "rank_compute": rank_count,
        "local_hbm": rank_count,
        "capacity_timeavg": rank_count,
        "hbm_watermark_summary": rank_count,
        "request": 1,
        "summary": 1,
        "consistency": 1,
    }
    for kind, expected in required_metric_counts.items():
        require(len(actual_metrics.get(kind, [])) == expected,
                f"full-detail metric {kind}: expected {expected} rows, got {len(actual_metrics.get(kind, []))}")
    consistencies = actual_metrics.get("consistency", [])
    require(len(consistencies) == 1 and consistencies[0].get("ok") is True and
            consistencies[0].get("violations") == [],
            "legacy metrics consistency record is not ok")
    summaries = actual_metrics.get("summary", [])
    require(len(summaries) == 1 and summaries[0].get("completed_unique_requests") == 1 and
            summaries[0].get("incomplete_requests") == 0 and summaries[0].get("sim_end_ns", 0) > 0,
            "full-detail summary has incomplete/invalid request or window totals")
    if summaries:
        sim_end = summaries[0]["sim_end_ns"]
        for kind in ("rank_compute", "local_hbm"):
            for record in actual_metrics.get(kind, []):
                require(record.get("utilization_window") == "[0, sim_end_ns]" and
                        record.get("utilization_window_ns") == sim_end,
                        f"metric {kind}: utilization gate/window differs from summary")
        for record in actual_metrics.get("capacity_timeavg", []):
            require(record.get("utilization_window") == "[0, sim_end_ns]" and
                    record.get("utilization_window_ns") == sim_end,
                    "metric capacity_timeavg: utilization gate/window differs from summary")
        for record in actual_metrics.get("rank_compute", []):
            coarse = record.get("coarse_die_runtime", {})
            require(coarse.get("utilization_window") == "[0, sim_end_ns]" and
                    coarse.get("utilization_window_ns") == sim_end,
                    f"rank {record.get('rank')}: coarse runtime window differs from summary")
        for record in actual_metrics.get("memory_anchor", []):
            require(isinstance(record.get("tick_ns"), int) and
                    0 <= record["tick_ns"] <= sim_end,
                    "memory_anchor.tick_ns is outside the simulation window")
        for record in actual_metrics.get("hbm_watermark_summary", []):
            if ("window_start_ns" in record or "window_end_ns" in record):
                require(record.get("window_start_ns") == 0 and
                        record.get("window_end_ns") == sim_end,
                        "HBM watermark summary window differs from simulation summary")
    require("gate=ok" in cpp_log,
            "C++ arrival/provenance gate did not report gate=ok")
    checks["metrics_contract"] = {
        "consistency_ok": bool(consistencies and consistencies[0].get("ok") is True),
        "sim_end_ns": summaries[0].get("sim_end_ns") if summaries else None,
        "gate_window_checked": bool(summaries),
        "observed_record_types": sorted(actual_metrics),
    }

report = {
    "status": "passed" if not errors else "failed",
    "auditor_sha256": __import__("hashlib").sha256(
        Path(__file__).read_bytes()).hexdigest(),
    "run_dir": str(RUN),
    "request_count": len(requests) if RUN.is_dir() and 'requests' in locals() else 0,
    "response_count": len(responses) if RUN.is_dir() and 'responses' in locals() else 0,
    "ack_count": len(acks) if RUN.is_dir() and 'acks' in locals() else 0,
    "checks": checks,
    "errors": errors,
}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(report, indent=2, sort_keys=True))
if errors:
    raise SystemExit(1)
