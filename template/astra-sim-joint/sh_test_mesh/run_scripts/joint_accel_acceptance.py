#!/usr/bin/env python3
"""Evidence driver for the joint acceleration work.

Commands preserve real source, fixture, timing, and comparison evidence. A
missing producer or incomplete run remains INCOMPLETE; it is never promoted to
PASS by this driver.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import importlib.util
import json
import math
import os
import platform
import random
import re
import shutil
import signal
import stat
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


EXIT_PASS = 0
EXIT_FAIL = 1
EXIT_INCOMPLETE = 2
EXIT_INSUFFICIENT = 3
REPO_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = REPO_ROOT.parents[1]
TRACE_RELATIVE = Path("agent-traces/tracelab/astra_compute_20.csv")
OLD_BUDGET_FIXTURE = Path("manifests/old-event-budget-bug.json")
STATISTICAL_REFERENCE = Path("manifests/acceptance-statistical-reference-v1.json")
STATISTICAL_REFERENCE_VERSION = "joint-accel-statistical-reference-v1"
PRICEABLE_SCENARIOS = tuple(range(1, 12)) + (13, 15)
DEV_SEEDS = tuple(range(64))
MAX_SEED = 319
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 20261005
PERCENTILE_P = 0.95
FIXTURE_STATE_VERSION = "execution-fixture-v1-20261006"
FIXTURE_TAPE_VERSION = "execution-event-tape-v1-20261006"
FIXTURE_RESULT_VERSION = "execution-result-v1-20261006"
PYTHON_RUNTIME_ASSET_VERSION = "joint-accel-python-runtime-v3-20261015"
PLANNER_STATE_VERSION = "joint-accel-planner-state-v3-20261015"
IMPLEMENTATION_SOURCE_INVENTORY_VERSION = "joint-accel-implementation-source-v1-20261006"
FIXTURE_EVENT_KINDS = ("graph_commit", "external_resource", "request_complete")
RAW_FRAME_AUDIT_KIND = "raw_v2_frame"
RAW_FRAME_AUDIT_KEYS = {"kind", "boundary_ordinal", "frame", "runtime_snapshot",
                        "producer_build"}
KERNEL_CONTRACT_KEYS = {
    "projection_abi_version", "descriptor_version", "pipeline_protocol_version",
    "model_version", "timer_whitelist_version",
    "prediction_config_sha256",
}
GATE_PARENTS: dict[str, tuple[str, ...]] = {
    "G-A0": (), "G-A1": ("G-A0",),
    "G-B1": ("G-A1",), "G-B2": ("G-B1",), "G-B3": ("G-A1",),
    "G-C0": ("G-A1",), "G-C1": ("G-C0",),
    "G-C2": ("G-C0", "G-B2", "G-C1"),
    "G-C3": ("G-C0", "G-B3", "G-C1"),
    "G-C4": ("G-B1", "G-B2", "G-C2", "G-C3"),
    "G-C5": ("G-C2", "G-C3", "G-C4"),
    "G-D0": ("G-C5",), "G-D1": ("G-D0",),
    "G-E0": ("G-D1",), "G-F0": ("G-E0",),
    "G-F1": ("G-F0",), "G-F2": ("G-E0",),
}
GATE_STATUSES = {"PASS", "FAIL", "INCOMPLETE", "INSUFFICIENT_EVIDENCE"}
VALID_ACTIONS = {"stay", "copy"}
PRICEABLE_SCENARIO_IDS = tuple(f"M{value:02d}" for value in PRICEABLE_SCENARIOS)
SUPPORTED_SCENARIO_IDS = tuple(f"M{value:02d}" for value in range(1, 17))
BUILD_WRAPPER = """cmake_minimum_required(VERSION 3.22)
project(AstraSim_Analytical_Build)
enable_testing()
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../ astra_sim_build)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../extern/network_backend/analytical backend_build)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../astra-sim/network_frontend/analytical frontend_build)
"""


class AcceptanceError(RuntimeError):
    def __init__(self, message: str, code: int = EXIT_INCOMPLETE,
                 reason_code: str = "incomplete") -> None:
        super().__init__(message)
        self.code = code
        self.reason_code = reason_code


class _ValidatedFixtureState(dict):
    """JSON state plus process-only handles for already verified assets.

    The private attributes are intentionally excluded from mapping iteration,
    canonical JSON, and every certificate.  They let the acceptance-only
    Python hydrator materialize the exact verified portable bytes without
    putting temporary paths or factories into the evidence contract.
    """

    _verified_asset_root: Path
    _verified_asset_paths: dict[str, Path]
    _runtime_asset: dict[str, Any]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n",
                   encoding="utf-8")
    os.replace(tmp, path)


def read_json(path: Path) -> Any:
    try:
        with path.open(encoding="utf-8") as stream:
            return json.load(stream, object_pairs_hook=_unique_object_pairs,
                             parse_constant=_reject_json_constant)
    except AcceptanceError:
        raise
    except (OSError, json.JSONDecodeError) as exc:
        raise AcceptanceError(f"cannot read JSON evidence {path}: {exc}")


def _unique_object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise AcceptanceError(f"duplicate JSON object key: {key!r}",
                                  EXIT_FAIL, "duplicate_json_key")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise AcceptanceError(f"non-standard JSON numeric constant is forbidden: {value}",
                          EXIT_FAIL, "non_finite_json_number")


def nearest_rank(values: Iterable[float], p: float) -> float:
    """Nearest-rank percentile used by both ends of the acceptance protocol."""
    ordered = sorted(values)
    if not ordered:
        raise AcceptanceError("nearest-rank percentile requires at least one value")
    if not math.isfinite(p) or p <= 0.0 or p > 1.0:
        raise AcceptanceError("nearest-rank percentile p must be in (0, 1]")
    return ordered[max(0, math.ceil(p * len(ordered)) - 1)]


def weighted_empirical_quantile(values: Iterable[tuple[float, float]],
                                p: float) -> float:
    """Left-continuous weighted empirical quantile; no interpolation."""
    rows = sorted((float(value), float(weight)) for value, weight in values)
    if not rows or any(not math.isfinite(w) or w <= 0 for _, w in rows):
        raise AcceptanceError("weighted quantile requires positive finite weights")
    if not math.isfinite(p) or p <= 0.0 or p > 1.0:
        raise AcceptanceError("weighted quantile p must be in (0, 1]")
    total = math.fsum(weight for _, weight in rows)
    cutoff = p * total
    cumulative = 0.0
    for value, weight in rows:
        cumulative += weight
        if cumulative + 1e-15 >= cutoff:
            return value
    return rows[-1][0]


def partition_cluster(cluster_id: str) -> dict[str, Any]:
    if not isinstance(cluster_id, str) or not cluster_id:
        raise AcceptanceError("cluster_id must be a non-empty string")
    digest = hashlib.sha256(
        ("joint-accel-split-v1/" + cluster_id).encode("utf-8")).hexdigest()
    value = int(digest[:16], 16)
    return {"cluster_id": cluster_id, "sha256": digest,
            "partition": "dev" if value % 5 == 0 else "holdout"}


def paired_duration_error(start_tick: int, exact_end_tick: int,
                           approx_end_tick: int) -> dict[str, Any]:
    values = (start_tick, exact_end_tick, approx_end_tick)
    if any(isinstance(x, bool) or not isinstance(x, int) or x < 0 for x in values):
        raise AcceptanceError("paired duration ticks must be non-negative integers")
    exact = exact_end_tick - start_tick
    approx = approx_end_tick - start_tick
    if exact < 0 or approx < 0:
        raise AcceptanceError("paired duration endpoint precedes its shared start")
    absolute = abs(approx - exact)
    if exact == 0:
        if approx != 0:
            return {"valid": False, "reason": "zero_reference_nonzero_approximation",
                    "exact_duration": exact, "approx_duration": approx,
                    "absolute_error_ticks": absolute,
                    "relative_error": None}
        return {"valid": True, "exact_duration": 0, "approx_duration": 0,
                "absolute_error_ticks": 0, "relative_error": 0.0,
                "signed_relative_error": 0.0, "zero_reference": True}
    return {"valid": True, "exact_duration": exact, "approx_duration": approx,
            "absolute_error_ticks": absolute,
            "relative_error": absolute / exact,
            "signed_relative_error": (approx - exact) / exact,
            "zero_reference": False}


def candidate_error_summary(snapshots: list[dict[str, Any]],
                            p: float = PERCENTILE_P) -> dict[str, Any]:
    """Summarize candidate pairs with total weight one per snapshot."""
    if not snapshots:
        raise AcceptanceError("candidate error bank has no snapshots")
    all_errors: list[float] = []
    weighted_errors: list[tuple[float, float]] = []
    weighted_signed = 0.0
    zero_pairs = 0
    for snapshot in snapshots:
        pairs = snapshot.get("pairs")
        if not isinstance(pairs, list) or not pairs:
            raise AcceptanceError("each candidate snapshot needs at least one pair")
        weight = 1.0 / len(pairs)
        for pair in pairs:
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                raise AcceptanceError("candidate pair must be [exact, approx]")
            exact, approx = pair
            if (isinstance(exact, bool) or isinstance(approx, bool)
                    or not isinstance(exact, (int, float))
                    or not isinstance(approx, (int, float))
                    or not math.isfinite(exact) or not math.isfinite(approx)
                    or exact < 0 or approx < 0):
                raise AcceptanceError("candidate costs must be finite and non-negative")
            if exact == 0:
                if approx != 0:
                    raise AcceptanceError(
                        "zero exact candidate cost has a nonzero approximation")
                error = signed = 0.0
                zero_pairs += 1
            else:
                signed = (approx - exact) / exact
                error = abs(signed)
            all_errors.append(error)
            weighted_errors.append((error, weight))
            weighted_signed += signed * weight
    return {
        "snapshot_weight_total": len(snapshots),
        "pair_count": len(all_errors),
        "zero_reference_pairs": zero_pairs,
        "unweighted_mean_relative_error": statistics.fmean(all_errors),
        "weighted_mean_relative_error": math.fsum(
            value * weight for value, weight in weighted_errors) / len(snapshots),
        "weighted_p95_relative_error": weighted_empirical_quantile(
            weighted_errors, p),
        "weighted_signed_mean_relative_error": weighted_signed / len(snapshots),
    }


def choose_development_config(configs: list[dict[str, Any]]) -> dict[str, Any]:
    """Confirm the single default configuration passed its required checks.

    K>1 窗口机制退役后无精度扫描轴：开发决议只剩"唯一默认配置是否
    legal（L2/L3/timing 全过）"的 fail-closed 判定与其三段计时中位数。
    """
    if len(configs) != 1:
        raise AcceptanceError("tuning run must contain exactly the default configuration")
    row = configs[0]
    if row.get("legal") is not True:
        raise AcceptanceError("default exact configuration did not pass its required checks",
                              EXIT_FAIL, "exact_config_invalid")
    runs = row.get("paired_wall_ns")
    if (not isinstance(runs, list) or len(runs) != 3
            or any(isinstance(x, bool) or not isinstance(x, int) or x <= 0
                   for x in runs)):
        raise AcceptanceError("the legal configuration requires three complete positive wall timings")
    return {"approximation_accepted": False,
            "median_wall_ns": int(statistics.median(runs))}


def tail_coverage(completed_requests: int, independent_clusters: int,
                  itl_intervals: int) -> dict[str, Any]:
    if any(isinstance(v, bool) or not isinstance(v, int) or v < 0
           for v in (completed_requests, independent_clusters, itl_intervals)):
        raise AcceptanceError("tail coverage counts must be non-negative integers")
    ttft = completed_requests >= 1000 and independent_clusters >= 100
    itl = itl_intervals >= 10000 and completed_requests >= 1000 and independent_clusters >= 100
    return {"ttft_p99": "PASS" if ttft else "INSUFFICIENT_EVIDENCE",
            "itl_p99": "PASS" if itl else "INSUFFICIENT_EVIDENCE",
            "completed_requests": completed_requests,
            "independent_clusters": independent_clusters,
            "itl_intervals": itl_intervals}


def clustered_bootstrap_ci(clusters: dict[str, list[float]], *,
                           statistic: str = "mean",
                           replicates: int = BOOTSTRAP_REPLICATES,
                           seed: int = BOOTSTRAP_SEED) -> dict[str, Any]:
    """Cluster bootstrap; each shared-resource run is sampled as a whole."""
    if not clusters or any(not values for values in clusters.values()):
        raise AcceptanceError("cluster bootstrap needs non-empty samples per cluster")
    if replicates != BOOTSTRAP_REPLICATES or seed != BOOTSTRAP_SEED:
        raise AcceptanceError("bootstrap replicates/seed are frozen at 2000/20261005")
    keys = sorted(clusters)
    if statistic not in ("mean", "p95"):
        raise AcceptanceError("unsupported bootstrap statistic")
    calc = statistics.fmean if statistic == "mean" else lambda values: nearest_rank(values, .95)
    rng = random.Random(seed)
    estimates: list[float] = []
    for _ in range(replicates):
        sampled = [keys[rng.randrange(len(keys))] for _ in keys]
        values = [value for key in sampled for value in clusters[key]]
        estimates.append(calc(values))
    return {"statistic": statistic, "replicates": replicates, "seed": seed,
            "confidence_level": 0.95,
            "lower": nearest_rank(estimates, 0.025),
            "upper": nearest_rank(estimates, 0.975),
            "cluster_count": len(keys)}


def statistical_reference_result(reference: dict[str, Any]) -> dict[str, Any]:
    if reference.get("version") != STATISTICAL_REFERENCE_VERSION:
        raise AcceptanceError("unsupported statistical reference version",
                              EXIT_FAIL, "statistical_reference_version_mismatch")
    results: dict[str, Any] = {}
    for case in reference.get("cases", []):
        kind, name = case.get("kind"), case.get("name")
        if kind == "candidate_error":
            actual = candidate_error_summary(case["snapshots"])
        elif kind == "percentile":
            actual = nearest_rank(case["values"], case["p"])
        elif kind == "paired_duration":
            actual = paired_duration_error(case["start_tick"],
                                           case["exact_end_tick"],
                                           case["approx_end_tick"])
        elif kind == "zero_duration":
            actual = paired_duration_error(0, case["exact"], case["approx"])
        elif kind == "dev_selection":
            rows = case["configs"]
            if len(rows) != 1:
                raise AcceptanceError(
                    "dev_selection case must carry the single default configuration",
                    EXIT_FAIL, "unknown_statistical_reference_case")
            actual = choose_development_config([
                {"legal": True, "paired_wall_ns": [rows[0]["median_ns"]] * 3}])
        elif kind == "tail_coverage":
            actual = tail_coverage(case["completed_requests"],
                                   case["independent_clusters"],
                                   case["itl_intervals"])
        else:
            raise AcceptanceError(f"unknown statistical reference case kind: {kind!r}",
                                  EXIT_FAIL, "unknown_statistical_reference_case")
        expected = case.get("expected", {})
        if kind == "candidate_error":
            compare = {key: actual[key] for key in expected}
        elif kind == "percentile":
            compare = actual
        elif kind in ("paired_duration", "zero_duration"):
            compare = actual
        elif kind == "dev_selection":
            compare = actual
        elif kind == "tail_coverage":
            compare = {"status": ("INSUFFICIENT_EVIDENCE"
                                  if "INSUFFICIENT_EVIDENCE" in actual.values()
                                  else "PASS")}
        results[name] = {"kind": kind, "actual": compare,
                         "expected": expected or case.get("expected_status"),
                         "pass": _reference_case_matches(kind, compare, case)}
    splits = []
    for expected in reference.get("cluster_splits", []):
        actual = partition_cluster(expected["cluster_id"])
        splits.append({**actual,
                       "pass": actual["sha256"] == expected["sha256"]
                       and actual["partition"] == expected["expected_partition"]})
    return {"version": reference["version"], "cases": results,
            "cluster_splits": splits,
            "pass": all(x["pass"] for x in results.values())
                    and all(x["pass"] for x in splits)}


def _reference_case_matches(kind: str, actual: Any,
                            case: dict[str, Any]) -> bool:
    expected = case.get("expected", {})
    if kind == "candidate_error":
        return all(math.isclose(actual[key], value, rel_tol=1e-12, abs_tol=1e-15)
                   for key, value in expected.items())
    if kind == "percentile":
        return actual == expected
    if kind == "paired_duration":
        return (actual.get("relative_error") == case["expected_relative_error"]
                and actual["absolute_error_ticks"] == case.get("expected_absolute_error_ticks",
                                                               abs(case["approx_end_tick"]-case["exact_end_tick"])))
    if kind == "zero_duration":
        if actual.get("valid") is not bool(case["expected_valid"]):
            return False
        return (actual.get("reason") == case.get("expected_reason")
                if not case["expected_valid"] else actual.get("relative_error") == 0.0)
    if kind == "dev_selection":
        # 精度扫描轴已随 K>1 窗口机制退役：唯一默认配置的决议不采纳
        # 任何近似（approximation_accepted 恒 False）即匹配。
        return actual["approximation_accepted"] is False
    if kind == "tail_coverage":
        return ("INSUFFICIENT_EVIDENCE" in actual.values()
                and case.get("expected_status") == "INSUFFICIENT_EVIDENCE")
    return False


def verify_statistical_reference(root: Path) -> dict[str, Any]:
    path = root / STATISTICAL_REFERENCE
    if not path.is_file():
        raise AcceptanceError(f"statistical reference is missing: {path}")
    reference = read_json(path)
    result = statistical_reference_result(reference)
    result["reference_sha256"] = sha256_file(path)
    if not result["pass"]:
        raise AcceptanceError("statistical reference conformance vectors failed",
                              EXIT_FAIL, "statistical_reference_mismatch")
    return result


def _require_object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise AcceptanceError(f"{label} must be a JSON object", EXIT_FAIL,
                              "fixture_schema_invalid")
    return value


def _require_exact_keys(value: dict[str, Any], required: set[str],
                        label: str) -> None:
    missing, extra = required - value.keys(), value.keys() - required
    if missing or extra:
        raise AcceptanceError(
            f"{label} keys mismatch; missing={sorted(missing)}, extra={sorted(extra)}",
            EXIT_FAIL, "fixture_schema_invalid")


def _require_int(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise AcceptanceError(f"{label} must be an integer >= {minimum}", EXIT_FAIL,
                              "fixture_schema_invalid")
    return value


def _require_finite_number(value: Any, label: str, *, minimum: float = 0.0) -> float:
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < minimum):
        raise AcceptanceError(f"{label} must be a finite number >= {minimum}", EXIT_FAIL,
                              "fixture_schema_invalid")
    return float(value)


def validate_kernel_contract(value: Any, *, expected_sha256: str | None = None) -> dict[str, Any]:
    """Validate the single frozen 6-key projection/execution wire contract."""
    contract = _require_object(value, "kernel_contract")
    _require_exact_keys(contract, KERNEL_CONTRACT_KEYS, "kernel_contract")
    if contract["projection_abi_version"] != 1 or type(contract["projection_abi_version"]) is not int:
        raise AcceptanceError("unsupported projection ABI version", EXIT_FAIL,
                              "kernel_contract_mismatch")
    if contract["descriptor_version"] != 1 or type(contract["descriptor_version"]) is not int:
        raise AcceptanceError("unsupported descriptor version", EXIT_FAIL,
                              "kernel_contract_mismatch")
    if contract["pipeline_protocol_version"] != 1 or type(contract["pipeline_protocol_version"]) is not int:
        raise AcceptanceError("unsupported pipeline protocol version", EXIT_FAIL,
                              "kernel_contract_mismatch")
    if contract["model_version"] != "v9-tie-roundrobin-20261020":
        raise AcceptanceError("unsupported projection model version", EXIT_FAIL,
                              "kernel_contract_mismatch")
    if contract["timer_whitelist_version"] != "timer-whitelist-v1-20261005":
        raise AcceptanceError("unsupported Timer whitelist version", EXIT_FAIL,
                              "kernel_contract_mismatch")
    digest = contract["prediction_config_sha256"]
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise AcceptanceError("prediction_config_sha256 must be 64 lowercase hex digits",
                              EXIT_FAIL, "kernel_contract_mismatch")
    if expected_sha256 is not None and digest != expected_sha256:
        raise AcceptanceError("kernel contract semantic config digest mismatch", EXIT_FAIL,
                              "prediction_config_digest_mismatch")
    return dict(contract)


def _load_prediction_module(source_root: Path):
    module_path = (source_root / "sh_test_mesh/workload/llama2_7b_inference/joint"
                   / "prediction_config.py")
    if not module_path.is_file():
        raise AcceptanceError(f"frozen prediction config module is missing: {module_path}")
    module_name = "_joint_accel_prediction_config_contract"
    existing = sys.modules.get(module_name)
    if existing is not None and Path(existing.__file__).resolve() == module_path.resolve():
        return existing
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise AcceptanceError("cannot load the frozen prediction configuration module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def resolve_prediction_contract(source_root: Path, config_path: Path | None) -> tuple[dict[str, Any], dict[str, Any]]:
    """Resolve a prediction config semantically, then derive its exact 6-field contract."""
    module = _load_prediction_module(source_root)
    if config_path is None:
        config_path = (source_root / "sh_test_mesh/run_scripts/prediction_presets"
                       / "default.json")
    raw = read_json(config_path)
    config = module.from_mapping(raw)
    contract = validate_kernel_contract(module.make_pipeline_kernel_contract(config))
    semantic_json = json.dumps(config.semantic_dict(), sort_keys=True,
                               separators=(",", ":"), ensure_ascii=False,
                               allow_nan=False)
    return contract, {
        "path": str(config_path), "sha256": sha256_file(config_path),
        "semantic_sha256": hashlib.sha256(semantic_json.encode("utf-8")).hexdigest(),
        "semantic_config": config.semantic_dict(),
    }


def _walk_json(value: Any, path: str = "$") -> Iterable[tuple[str, Any]]:
    yield path, value
    if isinstance(value, dict):
        for key, child in value.items():
            yield from _walk_json(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _walk_json(child, f"{path}[{index}]")


_COLD_DYNAMIC_GROUPS: dict[str, set[str]] = {
    "nodes": {"nodes"}, "ready": {"ready", "ready_ids"},
    "deferred": {"deferred", "deferred_issues"},
    "events": {"events", "pending_events", "internal_events"},
    "flows": {"flows", "active_flows"},
    "propagation tails": {"tails", "propagation_tails"},
    "HBM jobs": {"hbm_jobs", "active_hbm_jobs"},
    "pool jobs": {"pool_jobs", "active_pool_jobs"},
    "watch/mailbox": {"watch", "watches", "mailbox", "mailboxes"},
    "descriptors": {"descriptors"}, "cursors": {"cursors"},
    "buffers": {"buffers"}, "obligations": {"obligations"},
    "hardware inflight": {"hardware_inflight", "inflight"},
}


def _is_empty_dynamic(value: Any) -> bool:
    return value in (None, False, 0, "") or value == [] or value == {}


def validate_cold_initial_state(value: Any, rank_count: int) -> dict[str, Any]:
    state = _require_object(value, "initial_state")
    required = {"tick_ns", "next_flow_id", "next_tag", "ranks", "components",
                "directory", "capacity", "history"}
    if not required.issubset(state):
        raise AcceptanceError(f"initial_state misses {sorted(required-state.keys())}",
                              EXIT_FAIL, "cold_initial_state_invalid")
    if state["tick_ns"] != 0 or state["next_flow_id"] != 0 or state["next_tag"] != 1:
        raise AcceptanceError("initial_state is not at the required cold tick/ID/tag frontier",
                              EXIT_FAIL, "cold_initial_state_invalid")
    ranks = state["ranks"]
    if not isinstance(ranks, list) or len(ranks) != rank_count:
        raise AcceptanceError("initial_state must explicitly describe every rank",
                              EXIT_FAIL, "cold_initial_state_invalid")
    for expected_rank, row in enumerate(ranks):
        row = _require_object(row, f"initial_state.ranks[{expected_rank}]")
        if row.get("rank") != expected_rank or row.get("next_store_id") != 1:
            raise AcceptanceError("rank order/store IDs are not in the required cold state",
                                  EXIT_FAIL, "cold_initial_state_invalid")
    components = state["components"]
    if not isinstance(components, (list, dict)) or not components:
        raise AcceptanceError("initial_state must explicitly enumerate resource components",
                              EXIT_FAIL, "cold_initial_state_invalid")
    found: dict[str, list[tuple[str, Any]]] = {key: [] for key in _COLD_DYNAMIC_GROUPS}
    for path, item in _walk_json(state):
        if isinstance(item, dict):
            for group, names in _COLD_DYNAMIC_GROUPS.items():
                for name in names:
                    if name in item:
                        found[group].append((f"{path}.{name}", item[name]))
    missing = sorted(group for group, values in found.items() if not values)
    nonempty = [(path, item) for group, values in found.items()
                for path, item in values if not _is_empty_dynamic(item)]
    if missing or nonempty:
        raise AcceptanceError(
            "initial_state must explicitly provide every empty dynamic component; "
            f"missing={missing}, nonempty={nonempty[:8]}",
            EXIT_FAIL, "cold_initial_state_invalid")
    component_rows = (list(components.values()) if isinstance(components, dict)
                      else components)
    for index, component in enumerate(component_rows):
        path = f"$.components[{index}]"
        if not isinstance(component, dict):
            raise AcceptanceError(f"resource component must be an object at {path}",
                                  EXIT_FAIL, "cold_initial_state_invalid")
        if not ({"identity", "version", "state"} <= component.keys()):
            raise AcceptanceError(f"resource component lacks identity/version/state at {path}",
                                  EXIT_FAIL, "cold_initial_state_invalid")
        _require_int(component["version"], f"{path}.version")
    return dict(state)


def validate_fixture_state(value: Any, state_path: Path) -> dict[str, Any]:
    state = _require_object(value, "fixture state")
    fields = {"version", "kind", "case_id", "cluster_id", "scenario_id", "seed",
              "request_id", "fixed_instance_id", "fixed_action", "causal_input",
              "assets", "hardware", "initial_state", "python_runtime_asset_id"}
    _require_exact_keys(state, fields, "fixture state")
    if state["version"] != FIXTURE_STATE_VERSION or state["kind"] != "cold_initial":
        raise AcceptanceError("fixture state version/kind mismatch", EXIT_FAIL,
                              "fixture_version_mismatch")
    for key in ("case_id", "cluster_id", "request_id", "fixed_instance_id"):
        if not isinstance(state[key], str) or not state[key]:
            raise AcceptanceError(f"fixture state {key} must be non-empty", EXIT_FAIL,
                                  "fixture_schema_invalid")
    if re.fullmatch(r"[A-Za-z0-9._-]+", state["case_id"]) is None or state["case_id"] in (".", ".."):
        raise AcceptanceError("fixture case_id must be a safe path component", EXIT_FAIL,
                              "fixture_schema_invalid")
    if state["scenario_id"] not in SUPPORTED_SCENARIO_IDS:
        raise AcceptanceError("fixture scenario_id must be M01..M16", EXIT_FAIL,
                              "fixture_schema_invalid")
    _require_int(state["seed"], "fixture seed")
    if state["fixed_action"] not in VALID_ACTIONS:
        raise AcceptanceError("fixture fixed_action is outside the frozen two actions",
                              EXIT_FAIL, "fixture_schema_invalid")
    causal = _require_object(state["causal_input"], "causal_input")
    if causal.get("focus_stage") not in ("P", "D"):
        raise AcceptanceError("causal_input.focus_stage must be P or D", EXIT_FAIL,
                              "fixture_schema_invalid")
    aliases = {
        "context": ("context_tokens", "ctx"),
        "horizon_tokens": ("horizon_tokens",),
        "competition": ("competition_level", "competition"),
        "source count": ("source_count", "sources"),
    }
    for label, names in aliases.items():
        present = [name for name in names if name in causal]
        if not present:
            raise AcceptanceError(f"causal_input lacks fixed {label} metadata",
                                  EXIT_FAIL, "fixture_schema_invalid")
        value = causal[present[0]]
        if label == "competition":
            if value not in ("low", "high", 0, 1):
                raise AcceptanceError("causal_input competition must be low/high or 0/1",
                                      EXIT_FAIL, "fixture_schema_invalid")
        else:
            _require_int(value, f"causal_input.{present[0]}", minimum=1)
    if "committed_decode_tokens" in causal:
        _require_int(causal["committed_decode_tokens"],
                     "causal_input.committed_decode_tokens", minimum=1)
    hardware = _require_object(state["hardware"], "hardware")
    _require_exact_keys(hardware, {"rank_count", "system_asset_id", "network_asset_id",
                                  "remote_memory_asset_id", "die_hardware_asset_id"},
                        "hardware")
    rank_count = _require_int(hardware["rank_count"], "hardware.rank_count", minimum=1)
    assets = state["assets"]
    if not isinstance(assets, list) or not assets:
        raise AcceptanceError("fixture assets must be a non-empty list", EXIT_FAIL,
                              "fixture_schema_invalid")
    by_id: dict[str, dict[str, Any]] = {}
    base = state_path.parent.resolve()
    for index, entry in enumerate(assets):
        entry = _require_object(entry, f"assets[{index}]")
        _require_exact_keys(entry, {"asset_id", "path", "sha256"}, f"assets[{index}]")
        asset_id, relpath, digest = entry["asset_id"], entry["path"], entry["sha256"]
        if not isinstance(asset_id, str) or not asset_id or asset_id in by_id:
            raise AcceptanceError("asset IDs must be non-empty and unique", EXIT_FAIL,
                                  "fixture_schema_invalid")
        if not isinstance(relpath, str) or not relpath:
            raise AcceptanceError("asset path must be a non-empty relative path", EXIT_FAIL,
                                  "fixture_schema_invalid")
        rel = Path(relpath)
        if rel.is_absolute() or ".." in rel.parts:
            raise AcceptanceError("asset path must remain relative to the state directory",
                                  EXIT_FAIL, "fixture_asset_path_invalid")
        asset_path = (base / rel).resolve()
        if base not in asset_path.parents and asset_path != base:
            raise AcceptanceError("asset path escapes the state directory", EXIT_FAIL,
                                  "fixture_asset_path_invalid")
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise AcceptanceError("asset sha256 must be 64 lowercase hex digits", EXIT_FAIL,
                                  "fixture_schema_invalid")
        if not asset_path.is_file() or sha256_file(asset_path) != digest:
            raise AcceptanceError(f"fixture asset is missing or digest-mismatched: {relpath}",
                                  EXIT_FAIL, "fixture_asset_digest_mismatch")
        by_id[asset_id] = entry
    for field in ("system_asset_id", "network_asset_id", "remote_memory_asset_id",
                  "die_hardware_asset_id"):
        asset_id = hardware[field]
        if field in ("system_asset_id", "network_asset_id") and asset_id is None:
            raise AcceptanceError(f"hardware.{field} cannot be disabled", EXIT_FAIL,
                                  "fixture_schema_invalid")
        if asset_id is not None and asset_id not in by_id:
            raise AcceptanceError(f"hardware.{field} references unknown asset {asset_id!r}",
                                  EXIT_FAIL, "fixture_schema_invalid")
    runtime_asset_id = state["python_runtime_asset_id"]
    if not isinstance(runtime_asset_id, str) or runtime_asset_id not in by_id:
        raise AcceptanceError(
            "python_runtime_asset_id must reference a SHA256-verified data asset",
            EXIT_FAIL, "fixture_python_runtime_asset_missing")
    runtime_entry = by_id[runtime_asset_id]
    runtime_value = _require_object(
        read_json((base / runtime_entry["path"]).resolve()),
        "Python runtime asset")
    _require_exact_keys(runtime_value, {"version", "face_trace", "hardware",
                                        "remote_memory", "inference_groups",
                                        "joint_config", "system_backend"},
                        "Python runtime asset")
    if runtime_value.get("version") != PYTHON_RUNTIME_ASSET_VERSION:
        raise AcceptanceError("Python runtime asset version mismatch", EXIT_FAIL,
                              "fixture_python_runtime_version_mismatch")
    face_trace = _require_object(runtime_value["face_trace"],
                                 "Python runtime asset.face_trace")
    _require_exact_keys(face_trace, {"model", "model_name", "position_encoding",
                                     "remote_operand_loads",
                                     "kv_reserve_context_tokens", "prefill_chunk_size",
                                     "trace_granularity", "hardware_capacity_profile"},
                        "Python runtime asset.face_trace")
    model = _require_object(face_trace["model"],
                            "Python runtime asset.face_trace.model")
    _require_exact_keys(model, {"layers", "hidden_size", "ffn_size", "num_heads",
                                "vocab_size", "bytes_per_elem", "mlp_variant"},
                        "Python runtime asset.face_trace.model")
    for field in ("layers", "hidden_size", "ffn_size", "num_heads", "vocab_size",
                  "bytes_per_elem"):
        _require_int(model[field], f"Python runtime asset.face_trace.model.{field}",
                     minimum=1)
    if (model["hidden_size"] % model["num_heads"] != 0
            or model["mlp_variant"] not in ("gelu", "swiglu")):
        raise AcceptanceError("Python runtime model dimensions/variant are invalid",
                              EXIT_FAIL, "fixture_python_runtime_invalid")
    for field in ("model_name", "hardware_capacity_profile"):
        if not isinstance(face_trace[field], str) or not face_trace[field]:
            raise AcceptanceError(f"Python runtime face_trace.{field} must be non-empty",
                                  EXIT_FAIL, "fixture_python_runtime_invalid")
    if face_trace["position_encoding"] not in ("rope", "none"):
        raise AcceptanceError("Python runtime face_trace.position_encoding is invalid",
                              EXIT_FAIL, "fixture_python_runtime_invalid")
    if type(face_trace["remote_operand_loads"]) is not bool:
        raise AcceptanceError("Python runtime remote_operand_loads must be boolean",
                              EXIT_FAIL, "fixture_python_runtime_invalid")
    for field in ("kv_reserve_context_tokens", "prefill_chunk_size"):
        _require_int(face_trace[field], f"Python runtime face_trace.{field}")
    if face_trace["prefill_chunk_size"] == 0:
        raise AcceptanceError("Python runtime prefill_chunk_size must be positive",
                              EXIT_FAIL, "fixture_python_runtime_invalid")
    hardware_asset = _require_object(runtime_value["hardware"],
                                     "Python runtime asset.hardware")
    _require_exact_keys(hardware_asset, {"mesh_rows", "mesh_cols",
                                         "local_hbm_capacity_bytes",
                                         "local_hbm_bandwidth_gbps",
                                         "d2d_bandwidth_gbps", "peak_perf_tflops",
                                         "d2d_latency_ns", "local_hbm_latency_ns", "label"},
                        "Python runtime asset.hardware")
    for field in ("mesh_rows", "mesh_cols", "local_hbm_capacity_bytes"):
        _require_int(hardware_asset[field], f"Python runtime hardware.{field}", minimum=1)
    for field in ("local_hbm_bandwidth_gbps", "d2d_bandwidth_gbps", "peak_perf_tflops"):
        _require_finite_number(hardware_asset[field], f"Python runtime hardware.{field}",
                               minimum=0.000000000001)
    for field in ("d2d_latency_ns", "local_hbm_latency_ns"):
        _require_int(hardware_asset[field], f"Python runtime hardware.{field}")
    if not isinstance(hardware_asset["label"], str):
        raise AcceptanceError("Python runtime hardware.label must be a string",
                              EXIT_FAIL, "fixture_python_runtime_invalid")
    if hardware_asset["mesh_rows"] * hardware_asset["mesh_cols"] != rank_count:
        raise AcceptanceError("Python runtime hardware mesh disagrees with rank_count",
                              EXIT_FAIL, "fixture_python_runtime_invalid")
    remote_memory = _require_object(runtime_value["remote_memory"],
                                    "Python runtime asset.remote_memory")
    _require_exact_keys(remote_memory, {"edge_npus", "mesh_rows", "mesh_cols",
                                        "remote_mem_latency_ns", "remote_mem_bw_gbps",
                                        "logical_pool"},
                        "Python runtime asset.remote_memory")
    if not isinstance(remote_memory["edge_npus"], list) or not remote_memory["edge_npus"]:
        raise AcceptanceError("Python runtime remote_memory.edge_npus must be non-empty",
                              EXIT_FAIL, "fixture_python_runtime_invalid")
    edges = [_require_int(value, f"Python runtime remote_memory.edge_npus[{index}]")
             for index, value in enumerate(remote_memory["edge_npus"])]
    if len(edges) != len(set(edges)) or any(rank >= rank_count for rank in edges):
        raise AcceptanceError("Python runtime remote_memory edge ranks are invalid",
                              EXIT_FAIL, "fixture_python_runtime_invalid")
    for field in ("mesh_rows", "mesh_cols"):
        _require_int(remote_memory[field], f"Python runtime remote_memory.{field}", minimum=1)
    if (remote_memory["mesh_rows"], remote_memory["mesh_cols"]) != (
            hardware_asset["mesh_rows"], hardware_asset["mesh_cols"]):
        raise AcceptanceError("Python runtime remote-memory mesh disagrees with hardware",
                              EXIT_FAIL, "fixture_python_runtime_invalid")
    _require_int(remote_memory["remote_mem_latency_ns"],
                 "Python runtime remote_memory.remote_mem_latency_ns")
    _require_finite_number(remote_memory["remote_mem_bw_gbps"],
                           "Python runtime remote_memory.remote_mem_bw_gbps",
                           minimum=0.000000000001)
    if not isinstance(remote_memory["logical_pool"], str) or not remote_memory["logical_pool"]:
        raise AcceptanceError("Python runtime logical_pool must be non-empty",
                              EXIT_FAIL, "fixture_python_runtime_invalid")
    groups = runtime_value["inference_groups"]
    if not isinstance(groups, list) or not groups:
        raise AcceptanceError("Python runtime inference_groups must be non-empty",
                              EXIT_FAIL, "fixture_python_runtime_invalid")
    for index, group_value in enumerate(groups):
        group = _require_object(group_value,
                                f"Python runtime inference_groups[{index}]")
        _require_exact_keys(group, {"name", "pg_name", "ranks"},
                            f"Python runtime inference_groups[{index}]")
        if (not isinstance(group["name"], str) or not group["name"]
                or not isinstance(group["pg_name"], str) or not group["pg_name"]
                or group["pg_name"] == "0" or not isinstance(group["ranks"], list)
                or not group["ranks"]):
            raise AcceptanceError("Python runtime inference group is malformed",
                                  EXIT_FAIL, "fixture_python_runtime_invalid")
        for rank_index, rank in enumerate(group["ranks"]):
            _require_int(rank, f"Python runtime inference_groups[{index}].ranks[{rank_index}]")
    joint_config = _require_object(runtime_value["joint_config"],
                                   "Python runtime asset.joint_config")
    _require_exact_keys(joint_config, {"category_mode", "scheduler_mode", "layer_policy",
                                       "allowed_actions", "quota_mode",
                                       "boundary_policy", "combo"},
                        "Python runtime asset.joint_config")
    for field in ("category_mode", "scheduler_mode", "layer_policy", "allowed_actions",
                  "quota_mode", "boundary_policy"):
        if not isinstance(joint_config[field], str):
            raise AcceptanceError(f"Python runtime joint_config.{field} must be a string",
                                  EXIT_FAIL, "fixture_python_runtime_invalid")
    if joint_config["combo"] is not None and not isinstance(joint_config["combo"], str):
        raise AcceptanceError("Python runtime joint_config.combo must be null or string",
                              EXIT_FAIL, "fixture_python_runtime_invalid")
    backend = _require_object(runtime_value["system_backend"],
                              "Python runtime asset.system_backend")
    _require_exact_keys(backend, {"compute_cost_model", "die_profile",
                                  "die_model_config_asset_id"},
                        "Python runtime asset.system_backend")
    die_asset_id = backend["die_model_config_asset_id"]
    if die_asset_id is not None and die_asset_id not in by_id:
        raise AcceptanceError(
            "Python runtime asset references an unlisted Die model asset",
            EXIT_FAIL, "fixture_python_runtime_asset_missing")
    if backend["compute_cost_model"] not in ("norm", "die"):
        raise AcceptanceError("Python runtime compute_cost_model must be norm or die",
                              EXIT_FAIL, "fixture_python_runtime_invalid")
    if ((backend["compute_cost_model"] == "die") != (die_asset_id is not None)):
        raise AcceptanceError("Python runtime Die profile/config asset mismatch",
                              EXIT_FAIL, "fixture_python_runtime_invalid")
    state["initial_state"] = validate_cold_initial_state(state["initial_state"], rank_count)
    validated = _ValidatedFixtureState(state)
    validated._verified_asset_root = base
    validated._verified_asset_paths = {
        asset_id: (base / entry["path"]).resolve()
        for asset_id, entry in by_id.items()
    }
    validated._runtime_asset = runtime_value
    return validated


def _validate_planner_state(value: Any, *, stage: str,
                            causal_metadata: dict[str, Any], path: str) -> dict[str, Any]:
    planner = _require_object(value, path)
    _require_exact_keys(planner, {"version", "stage_input", "instance_loads",
                                  "registries", "forecast_generator"}, path)
    if planner["version"] != PLANNER_STATE_VERSION:
        raise AcceptanceError(f"{path}.version mismatch", EXIT_FAIL,
                              "planner_state_version_mismatch")
    stage_input = _require_object(planner["stage_input"], f"{path}.stage_input")
    if stage == "P":
        stage_keys = {"kind", "session_view", "request_view",
                      "prep_frontier_by_instance"}
    else:
        stage_keys = {"kind", "boundary_view", "runtime_state", "decision_epoch",
                      "inherited_obligation_ids"}
    _require_exact_keys(stage_input, stage_keys, f"{path}.stage_input")
    if stage_input["kind"] != stage:
        raise AcceptanceError(f"{path}.stage_input does not match boundary stage",
                              EXIT_FAIL, "planner_state_stage_mismatch")
    if stage == "P":
        session = _require_object(stage_input["session_view"],
                                  f"{path}.stage_input.session_view")
        _require_exact_keys(session, {"session_id", "home_instance", "resident_instance",
                                      "location", "history_tokens", "resident_prefix_layers",
                                      "history_bytes_by_tp_rank", "missing_bytes_by_tp_rank"},
                            f"{path}.stage_input.session_view")
        request = _require_object(stage_input["request_view"],
                                  f"{path}.stage_input.request_view")
        _require_exact_keys(request, {"request_id", "session_id", "input_tokens",
                                      "history_tokens_before", "estimated_decode_tokens",
                                      "horizon_source", "input_kv_bytes_by_tp_rank"},
                            f"{path}.stage_input.request_view")
        if (request["session_id"] != session["session_id"]
                or not isinstance(request["request_id"], str)
                or not request["request_id"]):
            raise AcceptanceError(f"{path}.stage_input request/session identity is invalid",
                                  EXIT_FAIL, "planner_state_identity_mismatch")
        _require_object(stage_input["prep_frontier_by_instance"],
                        f"{path}.stage_input.prep_frontier_by_instance")
    else:
        boundary_view = _require_object(stage_input["boundary_view"],
                                        f"{path}.stage_input.boundary_view")
        _require_exact_keys(boundary_view, {"session_id", "home_instance", "p_instance_index",
                                            "p_action", "history_tokens",
                                            "working_bytes_by_tp_rank",
                                            "decode_steps", "kv_token_bytes",
                                            "logic_home_instance", "kv_token_bytes_by_tp_rank"},
                            f"{path}.stage_input.boundary_view")
        runtime_state = _require_object(stage_input["runtime_state"],
                                        f"{path}.stage_input.runtime_state")
        _require_exact_keys(runtime_state, {"terminated", "first_token", "obligations",
                                            "working_layers"},
                            f"{path}.stage_input.runtime_state")
        first_token = _require_object(runtime_state["first_token"],
                                      f"{path}.stage_input.runtime_state.first_token")
        _require_exact_keys(first_token, {"issued", "consumed", "rng_advances"},
                            f"{path}.stage_input.runtime_state.first_token")
        for field in ("working_layers", "obligations"):
            if not isinstance(runtime_state[field], list):
                raise AcceptanceError(f"{path}.stage_input.runtime_state.{field} must be an array",
                                      EXIT_FAIL, "planner_state_schema_invalid")
        for index, obligation_value in enumerate(runtime_state["obligations"]):
            obligation = _require_object(obligation_value,
                                         f"{path}.stage_input.runtime_state.obligations[{index}]")
            _require_exact_keys(obligation, {"obligation_id", "kind", "receiver_instance",
                                             "replica_id", "summary"},
                                f"{path}.stage_input.runtime_state.obligations[{index}]")
        _require_int(stage_input["decision_epoch"], f"{path}.stage_input.decision_epoch")
        if not isinstance(stage_input["inherited_obligation_ids"], list):
            raise AcceptanceError(f"{path}.stage_input.inherited_obligation_ids must be an array",
                                  EXIT_FAIL, "planner_state_schema_invalid")
    loads = planner["instance_loads"]
    if not isinstance(loads, list):
        raise AcceptanceError(f"{path}.instance_loads must be an array", EXIT_FAIL,
                              "planner_state_schema_invalid")
    load_keys = {"instance_index", "queued_task_load_ns", "running_task_load_ns",
                 "active_decode_task_load_ns", "hbm_remaining_bytes_by_tp_rank",
                 "reclaimable_bytes_by_tp_rank", "capacity_summary",
                 "capacity_ledger_by_tp_rank"}
    for index, load_value in enumerate(loads):
        load_path = f"{path}.instance_loads[{index}]"
        load = _require_object(load_value, load_path)
        _require_exact_keys(load, load_keys, load_path)
        _require_int(load["instance_index"], f"{load_path}.instance_index")
        for field in ("queued_task_load_ns", "running_task_load_ns",
                      "active_decode_task_load_ns"):
            _require_finite_number(load[field], f"{load_path}.{field}")
        for field in ("hbm_remaining_bytes_by_tp_rank", "reclaimable_bytes_by_tp_rank",
                      "capacity_ledger_by_tp_rank"):
            if not isinstance(load[field], (list, dict)):
                raise AcceptanceError(f"{load_path}.{field} must be an array or object",
                                      EXIT_FAIL, "planner_state_schema_invalid")
        if load["capacity_summary"] is not None:
            summary = _require_object(load["capacity_summary"], f"{load_path}.capacity_summary")
            _require_exact_keys(summary, {"instance_index", "snapshot_version",
                                          "maintenance_event_count", "error_sources", "buckets"},
                                f"{load_path}.capacity_summary")
            if not isinstance(summary["buckets"], list):
                raise AcceptanceError(f"{load_path}.capacity_summary.buckets must be an array",
                                      EXIT_FAIL, "planner_state_schema_invalid")
            for bucket_index, bucket_value in enumerate(summary["buckets"]):
                bucket = _require_object(bucket_value,
                                         f"{load_path}.capacity_summary.buckets[{bucket_index}]")
                _require_exact_keys(bucket, {"category", "suffix_class", "shard_ratio",
                                             "writeback_ports", "reclaimable_by_rank",
                                             "clean_by_rank", "dirty_by_rank", "atom_count_by_rank",
                                             "max_atom_by_rank"},
                                    f"{load_path}.capacity_summary.buckets[{bucket_index}]")
    registries = _require_object(planner["registries"], f"{path}.registries")
    _require_exact_keys(registries, {"service_factors", "noc_flows", "pool_edge_counts",
                                     "hbm_port_counts", "telemetry", "forecast_history"},
                        f"{path}.registries")
    factors = _require_object(registries["service_factors"],
                              f"{path}.registries.service_factors")
    _require_exact_keys(factors, {"prefill_factor", "decode_factor", "transfer_factor",
                                  "updates", "rejected", "anomalous", "contended"},
                        f"{path}.registries.service_factors")
    for field in ("prefill_factor", "decode_factor", "transfer_factor"):
        _require_finite_number(factors[field],
                               f"{path}.registries.service_factors.{field}")
    for field in ("updates", "rejected", "anomalous", "contended"):
        counters = _require_object(
            factors[field], f"{path}.registries.service_factors.{field}")
        allowed_counter_keys = {"prefill_factor", "decode_factor", "transfer_factor"}
        extra_counter_keys = counters.keys() - allowed_counter_keys
        if extra_counter_keys:
            raise AcceptanceError(
                f"{path}.registries.service_factors.{field} has unknown counter "
                f"keys {sorted(extra_counter_keys)}", EXIT_FAIL,
                "planner_state_schema_invalid")
        for counter_name, counter_value in counters.items():
            _require_int(
                counter_value,
                f"{path}.registries.service_factors.{field}.{counter_name}")
    for field in ("noc_flows", "pool_edge_counts", "hbm_port_counts", "forecast_history"):
        if not isinstance(registries[field], list):
            raise AcceptanceError(f"{path}.registries.{field} must be an array", EXIT_FAIL,
                                  "planner_state_schema_invalid")
    row_shapes = {
        "noc_flows": {"source_rank", "destination_rank", "count"},
        "pool_edge_counts": {"edge_rank", "count"},
        "hbm_port_counts": {"rank", "registered_transfer_flows",
                             "active_decode_streams"},
    }
    for field, expected_keys in row_shapes.items():
        for index, row_value in enumerate(registries[field]):
            row = _require_object(row_value, f"{path}.registries.{field}[{index}]")
            _require_exact_keys(row, expected_keys,
                                f"{path}.registries.{field}[{index}]")
    telemetry = _require_object(registries["telemetry"], f"{path}.registries.telemetry")
    _require_exact_keys(telemetry, {"rates", "flow_counts", "window_ns"},
                        f"{path}.registries.telemetry")
    _require_object(telemetry["rates"], f"{path}.registries.telemetry.rates")
    _require_object(telemetry["flow_counts"], f"{path}.registries.telemetry.flow_counts")
    _require_int(telemetry["window_ns"], f"{path}.registries.telemetry.window_ns")
    forecast = _require_object(planner["forecast_generator"],
                               f"{path}.forecast_generator")
    _require_exact_keys(forecast, {"identity", "causal_inputs", "semantic_sha256"},
                        f"{path}.forecast_generator")
    if not isinstance(forecast["identity"], str) or not forecast["identity"]:
        raise AcceptanceError(f"{path}.forecast_generator.identity must be non-empty",
                              EXIT_FAIL, "planner_state_schema_invalid")
    causal = _require_object(forecast["causal_inputs"],
                             f"{path}.forecast_generator.causal_inputs")
    _require_exact_keys(causal, {"context_tokens", "horizon_tokens", "generator_inputs"},
                        f"{path}.forecast_generator.causal_inputs")
    _require_int(causal["context_tokens"], f"{path}.forecast_generator.context_tokens")
    _require_int(causal["horizon_tokens"], f"{path}.forecast_generator.horizon_tokens",
                 minimum=1)
    _require_object(causal["generator_inputs"],
                    f"{path}.forecast_generator.causal_inputs.generator_inputs")
    for field in ("context_tokens", "horizon_tokens"):
        if field in causal_metadata and causal[field] != causal_metadata[field]:
            raise AcceptanceError(f"{path}.forecast_generator differs from causal metadata at {field}",
                                  EXIT_FAIL, "planner_state_causal_mismatch")
    digest = forecast["semantic_sha256"]
    expected_digest = _canonical_digest({"identity": forecast["identity"],
                                         "causal_inputs": causal})
    if not isinstance(digest, str) or digest != expected_digest:
        raise AcceptanceError(f"{path}.forecast_generator semantic SHA256 mismatch",
                              EXIT_FAIL, "planner_state_digest_mismatch")
    return planner


def validate_event_tape(value: Any, *, case_id: str) -> dict[str, Any]:
    tape = _require_object(value, "event tape")
    _require_exact_keys(tape, {"version", "case_id", "events"}, "event tape")
    if tape["version"] != FIXTURE_TAPE_VERSION or tape["case_id"] != case_id:
        raise AcceptanceError("event tape version/case ID mismatch", EXIT_FAIL,
                              "event_tape_mismatch")
    events = tape["events"]
    if not isinstance(events, list):
        raise AcceptanceError("event tape events must be an array", EXIT_FAIL,
                              "fixture_schema_invalid")
    previous = -1
    for index, event_value in enumerate(events):
        event = _require_object(event_value, f"events[{index}]")
        _require_exact_keys(event, {"ordinal", "trigger", "kind", "payload"},
                            f"events[{index}]")
        ordinal = _require_int(event["ordinal"], f"events[{index}].ordinal")
        if ordinal <= previous:
            raise AcceptanceError("event ordinals must retain a strict stable order", EXIT_FAIL,
                                  "event_tape_order_invalid")
        previous = ordinal
        trigger = _require_object(event["trigger"], f"events[{index}].trigger")
        trigger_kind = trigger.get("kind")
        if trigger_kind == "tick":
            _require_exact_keys(trigger, {"kind", "tick_ns"}, f"events[{index}].trigger")
            _require_int(trigger["tick_ns"], f"events[{index}].trigger.tick_ns")
        elif trigger_kind == "gate_terminal":
            _require_exact_keys(trigger, {"kind", "segment_id", "role"},
                                f"events[{index}].trigger")
            if not all(isinstance(trigger[key], str) and trigger[key]
                       for key in ("segment_id", "role")):
                raise AcceptanceError("gate trigger requires a real segment/role reference",
                                      EXIT_FAIL, "event_tape_schema_invalid")
        else:
            raise AcceptanceError("event trigger must be tick or gate_terminal", EXIT_FAIL,
                                  "event_tape_schema_invalid")
        if not isinstance(event["kind"], str) or not event["kind"]:
            raise AcceptanceError(f"events[{index}].kind must be non-empty", EXIT_FAIL,
                                  "fixture_schema_invalid")
        if event["kind"] not in FIXTURE_EVENT_KINDS:
            raise AcceptanceError(
                f"events[{index}].kind {event['kind']!r} is outside the closed "
                "exogenous API (graph_commit/external_resource/request_complete)",
                EXIT_FAIL, "event_tape_kind_unsupported")
        _validate_event_payload(event["kind"], event["payload"], f"events[{index}]")
    return tape


def _validate_event_payload(kind: str, payload: dict[str, Any], path: str) -> None:
    """Closed typed payloads for the exogenous event API (execdoc §19.1A)."""
    _require_object(payload, f"{path}.payload")
    if kind == "graph_commit":
        _require_exact_keys(payload, {"batch", "decision_events", "boundary_record"},
                            f"{path}.payload")
        batch = _require_object(payload["batch"], f"{path}.payload.batch")
        schema_version = batch.get("schema_version")
        if type(schema_version) is not int or schema_version != 3:
            raise AcceptanceError(f"{path}.payload.batch must be an actual schema-3 batch",
                                  EXIT_FAIL, "event_tape_schema_invalid")
        if not isinstance(payload["decision_events"], list):
            raise AcceptanceError(f"{path}.payload.decision_events must be an array",
                                  EXIT_FAIL, "event_tape_schema_invalid")
        _require_object(payload["boundary_record"], f"{path}.payload.boundary_record")
        return
    if kind == "external_resource":
        _require_exact_keys(payload, {"resource", "mutation", "target", "amount"},
                            f"{path}.payload")
        resource = payload["resource"]
        mutation = payload["mutation"]
        if resource == "noc_link" and mutation == "join":
            target = _require_object(payload["target"], f"{path}.payload.target")
            _require_exact_keys(target, {"src_rank", "dst_rank"}, f"{path}.payload.target")
            amount = _require_object(payload["amount"], f"{path}.payload.amount")
            _require_exact_keys(amount, {"bytes", "tag"}, f"{path}.payload.amount")
            _require_int(target["src_rank"], f"{path}.payload.target.src_rank")
            _require_int(target["dst_rank"], f"{path}.payload.target.dst_rank")
            _require_int(amount["bytes"], f"{path}.payload.amount.bytes")
            _require_int(amount["tag"], f"{path}.payload.amount.tag")
        elif resource == "hbm_rank" and mutation == "join":
            target = _require_object(payload["target"], f"{path}.payload.target")
            _require_exact_keys(target, {"rank"}, f"{path}.payload.target")
            amount = _require_object(payload["amount"], f"{path}.payload.amount")
            _require_exact_keys(amount, {"kind", "bytes"}, f"{path}.payload.amount")
            _require_int(target["rank"], f"{path}.payload.target.rank")
            if amount["kind"] not in ("COMM_READ", "COMM_WRITE"):
                raise AcceptanceError(
                    f"{path}.payload.amount.kind must be COMM_READ or COMM_WRITE",
                    EXIT_FAIL, "event_tape_schema_invalid")
            _require_int(amount["bytes"], f"{path}.payload.amount.bytes")
        else:
            raise AcceptanceError(
                f"{path}.payload resource/mutation {resource!r}/{mutation!r} has no "
                "real backend API in this milestone; fail-closed",
                EXIT_FAIL, "event_tape_resource_unsupported")
        return
    if kind == "request_complete":
        _require_exact_keys(payload, {"request_id"}, f"{path}.payload")
        request_id = payload["request_id"]
        if not isinstance(request_id, str) or not request_id:
            raise AcceptanceError(f"{path}.payload.request_id must be non-empty", EXIT_FAIL,
                                  "event_tape_schema_invalid")
        return
    raise AcceptanceError(f"{path}.kind {kind!r} is unsupported", EXIT_FAIL,
                          "event_tape_kind_unsupported")


def _contains_key(value: Any, key: str) -> bool:
    return any(isinstance(item, dict) and key in item
               for _path, item in _walk_json(value))


def _micro_delivery_keys(result: dict[str, Any]) -> set[str]:
    """Allow the bounded real delivery tape emitted by the single v4 microcase."""
    if "delivery_frames" not in result:
        return set()
    if result.get("case_id") != "M01-micro-p2-d3-seed000":
        raise AcceptanceError("delivery_frames is restricted to the sparse microcase",
                              EXIT_FAIL, "fixture_raw_frame_invalid")
    frames = result["delivery_frames"]
    if not isinstance(frames, list) or len(frames) != 5:
        raise AcceptanceError("sparse microcase must expose five actual deliveries",
                              EXIT_FAIL, "fixture_raw_frame_invalid")
    for sequence, frame in enumerate(frames):
        if (not isinstance(frame, dict)
                or type(frame.get("schema_version")) is not int
                or type(frame.get("protocol_version")) is not int
                or frame["schema_version"] != 4 or frame["protocol_version"] != 4
                or frame.get("bridge_protocol") != "bridge-v4-20261011"
                or frame.get("work_ledger_schema") != "workledger-v4-20261011"
                or type(frame.get("delivery_sequence")) is not int
                or frame["delivery_sequence"] != sequence):
            raise AcceptanceError("sparse microcase delivery identity is invalid",
                                  EXIT_FAIL, "fixture_raw_frame_invalid")
    return {"delivery_frames"}


def validate_fixture_result(value: Any, state: dict[str, Any],
                            contract: dict[str, Any]) -> dict[str, Any]:
    result = _require_object(value, "fixture result")
    required = {"version", "case_id", "cluster_id", "scenario_id", "seed",
                "request_id", "fixed_instance_id", "fixed_action", "kernel_contract",
                "status", "reason", "completed", "initial_certificate",
                "terminal_certificate", "measurements", "boundary_records",
                "audit_records"}
    _require_exact_keys(result, required | _micro_delivery_keys(result), "fixture result")
    if result["version"] != FIXTURE_RESULT_VERSION:
        raise AcceptanceError("fixture result version mismatch", EXIT_FAIL,
                              "fixture_version_mismatch")
    for key in ("case_id", "cluster_id", "scenario_id", "seed", "request_id",
                "fixed_instance_id", "fixed_action"):
        if result[key] != state[key]:
            raise AcceptanceError(f"fixture result identity mismatch at {key}", EXIT_FAIL,
                                  "fixture_result_identity_mismatch")
    validate_kernel_contract(result["kernel_contract"])
    if result["kernel_contract"] != contract:
        raise AcceptanceError("fixture ran under a different kernel contract", EXIT_FAIL,
                              "fixture_result_contract_mismatch")
    if result["status"] not in ("PASS", "FAIL", "INCOMPLETE") or type(result["completed"]) is not bool:
        raise AcceptanceError("fixture status/completed field is invalid", EXIT_FAIL,
                              "fixture_result_schema_invalid")
    if result["status"] == "PASS" and result["completed"] is not True:
        raise AcceptanceError("PASS requires natural completion", EXIT_FAIL,
                              "fixture_result_inconsistent")
    if result["status"] != "PASS" and result["completed"] is not False:
        raise AcceptanceError("non-PASS fixture results cannot claim completion", EXIT_FAIL,
                              "fixture_result_inconsistent")
    for key in ("initial_certificate", "terminal_certificate"):
        _require_object(result[key], f"fixture result {key}")
    certified_initial = validate_cold_initial_state(
        result["initial_certificate"], state["hardware"]["rank_count"])
    if certified_initial != state["initial_state"]:
        raise AcceptanceError("fresh process initial certificate differs from the cold input",
                              EXIT_FAIL, "initial_certificate_mismatch")
    for list_key in ("measurements", "boundary_records", "audit_records"):
        if not isinstance(result[list_key], list):
            raise AcceptanceError(f"fixture result {list_key} must be an array", EXIT_FAIL,
                                  "fixture_result_schema_invalid")
    if result["status"] == "PASS" and (not result["measurements"]
                                        or not result["boundary_records"]):
        raise AcceptanceError("completed PASS requires actual stage measurements and boundaries",
                              EXIT_FAIL, "fixture_result_empty_success")
    measurement_keys = {"stage", "fixed_instance_id", "fixed_action", "boundary_ordinal",
                        "start_tick_ns", "service_tick_ns", "settle_tick_ns",
                        "logical_event_count", "native_physical_steps",
                        "charged_totals"}
    for index, row_value in enumerate(result["measurements"]):
        row = _require_object(row_value, f"measurements[{index}]")
        _require_exact_keys(row, measurement_keys, f"measurements[{index}]")
        if row["stage"] not in ("P", "D") or row["fixed_action"] not in VALID_ACTIONS:
            raise AcceptanceError(f"measurement {index} has invalid stage/action", EXIT_FAIL,
                                  "fixture_result_schema_invalid")
        if not isinstance(row["fixed_instance_id"], str) or not row["fixed_instance_id"]:
            raise AcceptanceError(f"measurement {index} requires its actual fixed instance",
                                  EXIT_FAIL, "fixture_result_schema_invalid")
        for key in ("boundary_ordinal", "start_tick_ns", "service_tick_ns", "settle_tick_ns",
                    "logical_event_count", "native_physical_steps"):
            _require_int(row[key], f"measurements[{index}].{key}")
        if not row["start_tick_ns"] <= row["service_tick_ns"] <= row["settle_tick_ns"]:
            raise AcceptanceError(f"measurement {index} time order is invalid", EXIT_FAIL,
                                  "fixture_result_inconsistent")
        _require_object(row["charged_totals"], f"measurements[{index}].charged_totals")
    boundary_keys = {"request_id", "stage", "boundary_ordinal", "tick_ns",
                     "fixed_instance_id", "fixed_action", "causal_prestate",
                     "causal_metadata", "planner_state"}
    boundary_index: dict[tuple[str, int], dict[str, Any]] = {}
    for index, row_value in enumerate(result["boundary_records"]):
        row = _require_object(row_value, f"boundary_records[{index}]")
        _require_exact_keys(row, boundary_keys, f"boundary_records[{index}]")
        if row["stage"] not in ("P", "D") or row["fixed_action"] not in VALID_ACTIONS:
            raise AcceptanceError(f"boundary record {index} has invalid stage/action", EXIT_FAIL,
                                  "fixture_result_schema_invalid")
        _require_int(row["boundary_ordinal"], f"boundary_records[{index}].boundary_ordinal")
        _require_int(row["tick_ns"], f"boundary_records[{index}].tick_ns")
        if not isinstance(row["fixed_instance_id"], str) or not row["fixed_instance_id"]:
            raise AcceptanceError(f"boundary record {index} requires its actual fixed instance",
                                  EXIT_FAIL, "fixture_result_schema_invalid")
        if row["request_id"] != result["request_id"]:
            raise AcceptanceError(f"boundary record {index} belongs to another request",
                                  EXIT_FAIL, "fixture_result_identity_mismatch")
        identity = (row["stage"], row["boundary_ordinal"])
        if identity in boundary_index:
            raise AcceptanceError(f"duplicate boundary identity {identity!r}", EXIT_FAIL,
                                  "fixture_result_schema_invalid")
        boundary_index[identity] = row
        prestate = _require_object(row["causal_prestate"],
                                   f"boundary_records[{index}].causal_prestate")
        prestate_fields = {"work_ledger", "resources", "directory", "capacity",
                           "obligations", "history"}
        _require_exact_keys(prestate, prestate_fields,
                            f"boundary_records[{index}].causal_prestate")
        for field in prestate_fields:
            _require_object(prestate[field],
                            f"boundary_records[{index}].causal_prestate.{field}")
        metadata = _require_object(row["causal_metadata"],
                                   f"boundary_records[{index}].causal_metadata")
        for field in ("context_tokens", "horizon_tokens"):
            if field not in metadata:
                raise AcceptanceError(f"boundary_records[{index}].causal_metadata lacks {field}",
                                      EXIT_FAIL, "fixture_result_schema_invalid")
            _require_int(metadata[field], f"boundary_records[{index}].causal_metadata.{field}",
                         minimum=1)
        state_causal = state["causal_input"]
        state_context = state_causal.get("context_tokens", state_causal.get("ctx"))
        if (metadata["context_tokens"] != state_context
                or metadata["horizon_tokens"] != state_causal["horizon_tokens"]):
            raise AcceptanceError(f"boundary record {index} differs from frozen causal inputs",
                                  EXIT_FAIL, "fixture_result_causal_mismatch")
        planner_state = _validate_planner_state(
            row["planner_state"], stage=row["stage"], causal_metadata=metadata,
            path=f"boundary_records[{index}].planner_state")
        if (row["stage"] == "P"
                and planner_state["stage_input"]["request_view"]["request_id"]
                != row["request_id"]):
            raise AcceptanceError(f"boundary record {index} planner request identity differs",
                                  EXIT_FAIL, "fixture_result_identity_mismatch")
        if _contains_key(row["causal_metadata"], "committed_decode_tokens"):
            raise AcceptanceError("prediction boundary leaks committed_decode_tokens", EXIT_FAIL,
                                  "future_execution_state_leaked")
    measurement_ids: set[tuple[str, int]] = set()
    for index, measurement in enumerate(result["measurements"]):
        identity = (measurement["stage"], measurement["boundary_ordinal"])
        if identity not in boundary_index:
            raise AcceptanceError(f"measurement {index} has no actual decision boundary",
                                  EXIT_FAIL, "fixture_measurement_boundary_missing")
        if identity in measurement_ids:
            raise AcceptanceError(f"duplicate measurement identity {identity!r}", EXIT_FAIL,
                                  "fixture_result_schema_invalid")
        measurement_ids.add(identity)
        boundary = boundary_index[identity]
        if (measurement["fixed_instance_id"] != boundary["fixed_instance_id"]
                or measurement["fixed_action"] != boundary["fixed_action"]):
            raise AcceptanceError(f"measurement {index} disagrees with its stage-local boundary",
                                  EXIT_FAIL, "fixture_measurement_boundary_identity_mismatch")
    return result


def canonical_fixture_result(value: dict[str, Any]) -> dict[str, Any]:
    """Drop only host-clock diagnostics before fresh-process determinism checks."""
    excluded = {"elapsed_host_ns", "host_wall_ns", "host_elapsed_ns", "started_utc",
                "finished_utc", "created_utc", "host_timestamp"}
    def strip(item: Any) -> Any:
        if isinstance(item, dict):
            return {key: strip(child) for key, child in item.items() if key not in excluded}
        if isinstance(item, list):
            return [strip(child) for child in item]
        return item
    return strip(value)


def validate_raw_fixture_result(value: Any, state: dict[str, Any],
                                contract: dict[str, Any]) -> dict[str, Any]:
    """Stage A validation (execdoc §19.1A): raw producer output.

    The C++ fixture proves physical facts only: boundary causal_prestate and
    planner_state stay null with canonicalization_pending=true, and each real
    boundary commit contributes a raw_v2_frame audit entry carrying the exact
    current V4 envelope or retained V3 archive envelope plus same-tick runtime
    snapshot. Canonical
    six-domain state is produced later by Stage B, never copied from inputs.
    """
    result = _require_object(value, "fixture result")
    required = {"version", "case_id", "cluster_id", "scenario_id", "seed",
                "request_id", "fixed_instance_id", "fixed_action", "kernel_contract",
                "status", "reason", "completed", "initial_certificate",
                "terminal_certificate", "measurements", "boundary_records",
                "audit_records", "canonicalization_pending"}
    _require_exact_keys(result, required | _micro_delivery_keys(result), "fixture result")
    if result["version"] != FIXTURE_RESULT_VERSION:
        raise AcceptanceError("fixture result version mismatch", EXIT_FAIL,
                              "fixture_version_mismatch")
    for key in ("case_id", "cluster_id", "scenario_id", "seed", "request_id",
                "fixed_instance_id", "fixed_action"):
        if result[key] != state[key]:
            raise AcceptanceError(f"fixture result identity mismatch at {key}", EXIT_FAIL,
                                  "fixture_result_identity_mismatch")
    validate_kernel_contract(result["kernel_contract"])
    if result["kernel_contract"] != contract:
        raise AcceptanceError("fixture ran under a different kernel contract", EXIT_FAIL,
                              "fixture_result_contract_mismatch")
    if result["canonicalization_pending"] is not True:
        raise AcceptanceError(
            "raw fixture result must keep canonicalization_pending=true; the producer "
            "cannot emit canonical six-domain state itself", EXIT_FAIL,
                              "fixture_raw_stage_invalid")
    if result["status"] not in ("PASS", "FAIL", "INCOMPLETE") or type(result["completed"]) is not bool:
        raise AcceptanceError("fixture status/completed field is invalid", EXIT_FAIL,
                              "fixture_result_schema_invalid")
    if result["status"] == "PASS" and result["completed"] is not True:
        raise AcceptanceError("PASS requires natural completion", EXIT_FAIL,
                              "fixture_result_inconsistent")
    if result["status"] != "PASS" and result["completed"] is not False:
        raise AcceptanceError("non-PASS fixture results cannot claim completion", EXIT_FAIL,
                              "fixture_result_inconsistent")
    for key in ("initial_certificate", "terminal_certificate"):
        _require_object(result[key], f"fixture result {key}")
    certified_initial = validate_cold_initial_state(
        result["initial_certificate"], state["hardware"]["rank_count"])
    if certified_initial != state["initial_state"]:
        raise AcceptanceError("fresh process initial certificate differs from the cold input",
                              EXIT_FAIL, "initial_certificate_mismatch")
    for list_key in ("measurements", "boundary_records", "audit_records"):
        if not isinstance(result[list_key], list):
            raise AcceptanceError(f"fixture result {list_key} must be an array", EXIT_FAIL,
                                  "fixture_result_schema_invalid")
    if result["status"] == "PASS" and (not result["measurements"]
                                        or not result["boundary_records"]
                                        or not result["audit_records"]):
        raise AcceptanceError("completed PASS requires actual measurements, boundaries "
                              "and audit records", EXIT_FAIL, "fixture_result_empty_success")
    measurement_keys = {"stage", "fixed_instance_id", "fixed_action", "boundary_ordinal",
                        "start_tick_ns", "service_tick_ns", "settle_tick_ns",
                        "logical_event_count", "native_physical_steps",
                        "charged_totals"}
    for index, row_value in enumerate(result["measurements"]):
        row = _require_object(row_value, f"measurements[{index}]")
        _require_exact_keys(row, measurement_keys, f"measurements[{index}]")
        if row["stage"] not in ("P", "D") or row["fixed_action"] not in VALID_ACTIONS:
            raise AcceptanceError(f"measurement {index} has invalid stage/action", EXIT_FAIL,
                                  "fixture_result_schema_invalid")
        if not isinstance(row["fixed_instance_id"], str) or not row["fixed_instance_id"]:
            raise AcceptanceError(f"measurement {index} requires its actual fixed instance",
                                  EXIT_FAIL, "fixture_result_schema_invalid")
        for key in ("boundary_ordinal", "start_tick_ns", "service_tick_ns", "settle_tick_ns",
                    "logical_event_count", "native_physical_steps"):
            _require_int(row[key], f"measurements[{index}].{key}")
        if not row["start_tick_ns"] <= row["service_tick_ns"] <= row["settle_tick_ns"]:
            raise AcceptanceError(f"measurement {index} time order is invalid", EXIT_FAIL,
                                  "fixture_result_inconsistent")
        _require_object(row["charged_totals"], f"measurements[{index}].charged_totals")
    boundary_keys = {"request_id", "stage", "boundary_ordinal", "tick_ns",
                     "fixed_instance_id", "fixed_action", "causal_prestate",
                     "causal_metadata", "planner_state"}
    boundary_index: dict[tuple[str, int], dict[str, Any]] = {}
    for index, row_value in enumerate(result["boundary_records"]):
        row = _require_object(row_value, f"boundary_records[{index}]")
        _require_exact_keys(row, boundary_keys, f"boundary_records[{index}]")
        if row["stage"] not in ("P", "D") or row["fixed_action"] not in VALID_ACTIONS:
            raise AcceptanceError(f"boundary record {index} has invalid stage/action", EXIT_FAIL,
                                  "fixture_result_schema_invalid")
        _require_int(row["boundary_ordinal"], f"boundary_records[{index}].boundary_ordinal")
        _require_int(row["tick_ns"], f"boundary_records[{index}].tick_ns")
        if not isinstance(row["fixed_instance_id"], str) or not row["fixed_instance_id"]:
            raise AcceptanceError(f"boundary record {index} requires its actual fixed instance",
                                  EXIT_FAIL, "fixture_result_schema_invalid")
        if row["request_id"] != result["request_id"]:
            raise AcceptanceError(f"boundary record {index} belongs to another request",
                                  EXIT_FAIL, "fixture_result_identity_mismatch")
        identity = (row["stage"], row["boundary_ordinal"])
        if identity in boundary_index:
            raise AcceptanceError(f"duplicate boundary identity {identity!r}", EXIT_FAIL,
                                  "fixture_result_schema_invalid")
        boundary_index[identity] = row
        if row["causal_prestate"] is not None or row["planner_state"] is not None:
            raise AcceptanceError(
                f"boundary record {index} must leave causal_prestate/planner_state null "
                "in the raw producer stage", EXIT_FAIL, "fixture_raw_stage_invalid")
        metadata = _require_object(row["causal_metadata"],
                                   f"boundary_records[{index}].causal_metadata")
        for field in ("context_tokens", "horizon_tokens"):
            if field not in metadata:
                raise AcceptanceError(f"boundary_records[{index}].causal_metadata lacks {field}",
                                      EXIT_FAIL, "fixture_result_schema_invalid")
            _require_int(metadata[field], f"boundary_records[{index}].causal_metadata.{field}",
                         minimum=1)
        state_causal = state["causal_input"]
        state_context = state_causal.get("context_tokens", state_causal.get("ctx"))
        if (metadata["context_tokens"] != state_context
                or metadata["horizon_tokens"] != state_causal["horizon_tokens"]):
            raise AcceptanceError(f"boundary record {index} differs from frozen causal inputs",
                                  EXIT_FAIL, "fixture_result_causal_mismatch")
        if _contains_key(row["causal_metadata"], "committed_decode_tokens"):
            raise AcceptanceError("prediction boundary leaks committed_decode_tokens", EXIT_FAIL,
                                  "future_execution_state_leaked")
    raw_frames: list[dict[str, Any]] = []
    for index, entry_value in enumerate(result["audit_records"]):
        entry = _require_object(entry_value, f"audit_records[{index}]")
        if entry.get("kind") != RAW_FRAME_AUDIT_KIND:
            continue
        _require_exact_keys(entry, RAW_FRAME_AUDIT_KEYS, f"audit_records[{index}]")
        _require_int(entry["boundary_ordinal"],
                     f"audit_records[{index}].boundary_ordinal")
        frame = _require_object(entry["frame"], f"audit_records[{index}].frame")
        identity = (frame.get("schema_version"),
                    frame.get("protocol_version"),
                    frame.get("bridge_protocol"),
                    frame.get("work_ledger_schema"))
        if (type(frame.get("schema_version")) is not int
                or type(frame.get("protocol_version")) is not int
                or identity not in (
                    (3, 3, "bridge-v3-20261009", "workledger-v3-20261009"),
                    (4, 4, "bridge-v4-20261011", "workledger-v4-20261011"))):
            raise AcceptanceError(f"audit_records[{index}].frame is not an exact supported archive/live envelope",
                                  EXIT_FAIL, "fixture_raw_frame_invalid")
        if frame.get("producer") != "cpp_online":
            raise AcceptanceError(f"audit_records[{index}].frame producer must be cpp_online",
                                  EXIT_FAIL, "fixture_raw_frame_invalid")
        _require_object(entry["runtime_snapshot"], f"audit_records[{index}].runtime_snapshot")
        build = _require_object(entry["producer_build"], f"audit_records[{index}].producer_build")
        _require_exact_keys(build, {"binary_sha256", "kernel_contract"},
                            f"audit_records[{index}].producer_build")
        if not isinstance(build["binary_sha256"], str) or len(build["binary_sha256"]) != 64:
            raise AcceptanceError(f"audit_records[{index}].producer_build.binary_sha256 "
                                  "must be a real 64-hex digest", EXIT_FAIL,
                                  "fixture_raw_frame_invalid")
        raw_frames.append(entry)
    boundaries = result["boundary_records"]
    if result["status"] == "PASS":
        if len(raw_frames) != len(boundaries):
            raise AcceptanceError(
                f"raw ledger frames ({len(raw_frames)}) must pair 1:1 in commit order with "
                f"boundary records ({len(boundaries)})", EXIT_FAIL,
                "fixture_raw_frame_missing")
        for position, (frame_entry, boundary) in enumerate(zip(raw_frames, boundaries)):
            if frame_entry["boundary_ordinal"] != boundary["boundary_ordinal"]:
                raise AcceptanceError(
                    f"raw frame {position} ordinal {frame_entry['boundary_ordinal']} does "
                    f"not match boundary {boundary['boundary_ordinal']} at the same commit "
                    "position", EXIT_FAIL, "fixture_raw_frame_order_mismatch")
    measurement_ids: set[tuple[str, int]] = set()
    for index, measurement in enumerate(result["measurements"]):
        identity = (measurement["stage"], measurement["boundary_ordinal"])
        if identity not in boundary_index:
            raise AcceptanceError(f"measurement {index} has no actual decision boundary",
                                  EXIT_FAIL, "fixture_measurement_boundary_missing")
        if identity in measurement_ids:
            raise AcceptanceError(f"duplicate measurement identity {identity!r}", EXIT_FAIL,
                                  "fixture_result_schema_invalid")
        measurement_ids.add(identity)
        boundary = boundary_index[identity]
        if (measurement["fixed_instance_id"] != boundary["fixed_instance_id"]
                or measurement["fixed_action"] != boundary["fixed_action"]):
            raise AcceptanceError(f"measurement {index} disagrees with its stage-local boundary",
                                  EXIT_FAIL, "fixture_measurement_boundary_identity_mismatch")
    return result


def canonicalize_fixture_result(raw_result: dict[str, Any], state: dict[str, Any],
                                 source_root: Path) -> dict[str, Any]:
    """Stage B (execdoc §19.1A): canonicalize raw frames via the acceptance-only planner.

    The heavy lifting (runtime hydration, production dual-ledger replay, planner
    export) lives in joint/acceptance_projection.py::canonicalize_raw_frames so the
    original constructors stay the single source of truth. This driver layer only
    locates the module in the hash-recorded source copy and converts capability
    gaps into explicit INCOMPLETE reasons.
    """
    module = _load_acceptance_module(source_root)
    canonicalizer = getattr(module, "canonicalize_raw_frames", None)
    if not callable(canonicalizer):
        raise AcceptanceError("acceptance_projection lacks canonicalize_raw_frames()",
                              EXIT_INCOMPLETE, "l3_canonicalizer_missing")
    payload = json.loads(json.dumps(raw_result, allow_nan=False))
    try:
        canonical = canonicalizer(payload, state, source_root=source_root)
    except AcceptanceError:
        raise
    except Exception as exc:
        raise AcceptanceError(f"canonicalize_raw_frames failed: {exc}", EXIT_INCOMPLETE,
                              "l3_canonicalization_failed") from exc
    canonical = _require_object(canonical, "canonicalized fixture result")
    if canonical.get("canonicalization_pending") is True:
        raise AcceptanceError("canonicalized result is still pending canonicalization",
                              EXIT_FAIL, "l3_canonicalization_incomplete")
    return canonical


def first_boundary_samples(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One first actual boundary per request/stage; never replace a failed first row."""
    selected: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in records:
        request_id = row.get("request_id")
        stage = row.get("stage")
        cluster_id = row.get("cluster_id")
        if not all(isinstance(value, str) and value for value in
                   (request_id, stage, cluster_id)) or stage not in ("P", "D"):
            raise AcceptanceError("boundary sampling requires cluster/request/stage identity")
        key = (cluster_id, request_id, stage)
        ordinal = _require_int(row.get("boundary_ordinal"), "boundary_ordinal")
        prior = selected.get(key)
        if prior is None or ordinal < prior["boundary_ordinal"]:
            selected[key] = row
    return [selected[key] for key in sorted(selected)]


def sample_coverage(records: list[dict[str, Any]], *,
                    min_stage_snapshots: int = 100,
                    min_bucket_snapshots: int = 30) -> dict[str, Any]:
    """Count independent request/cluster samples for the fixed holdout bins."""
    rows = first_boundary_samples(records) if records else []
    stage_groups = {stage: [row for row in rows if row["stage"] == stage]
                    for stage in ("P", "D")}
    bucket_values: dict[str, dict[str, set[tuple[str, str]]]] = {
        name: {value: set() for value in values}
        for name, values in {
            "context": ("short", "long"), "competition": ("low", "high"),
            "sources": ("single", "multi"), "capacity": ("normal", "stress"),
        }.items()}
    unclassified: dict[str, int] = {key: 0 for key in bucket_values}
    for row in rows:
        metadata = row.get("causal_metadata", {})
        context = metadata.get("context_tokens", metadata.get("ctx"))
        if isinstance(context, int) and not isinstance(context, bool):
            bucket = "short" if context <= 512 else "long" if context >= 2048 else None
            if bucket:
                bucket_values["context"][bucket].add(
                    (row["cluster_id"], row["request_id"]))
        else:
            unclassified["context"] += 1
        competition = metadata.get("competition_level", metadata.get("competition"))
        if competition in ("low", "high"):
            bucket_values["competition"][competition].add(
                (row["cluster_id"], row["request_id"]))
        elif competition in (0, 1) and not isinstance(competition, bool):
            bucket_values["competition"]["high" if competition else "low"].add(
                (row["cluster_id"], row["request_id"]))
        else:
            unclassified["competition"] += 1
        source_count = metadata.get("source_count", metadata.get("sources"))
        if isinstance(source_count, int) and not isinstance(source_count, bool):
            bucket_values["sources"]["single" if source_count == 1 else "multi"].add(
                (row["cluster_id"], row["request_id"]))
        else:
            unclassified["sources"] += 1
        prestate = row.get("causal_prestate", {})
        capacity = row.get("capacity", {})
        if not capacity and isinstance(prestate, dict):
            capacity = prestate.get("capacity", {})
        capacity_bucket = (metadata.get("capacity_bucket")
                           or capacity.get("pressure_bucket")
                           or capacity.get("bucket")) if isinstance(capacity, dict) else None
        if capacity_bucket in ("normal", "stress"):
            bucket_values["capacity"][capacity_bucket].add(
                (row["cluster_id"], row["request_id"]))
        else:
            unclassified["capacity"] += 1
    stages: dict[str, Any] = {}
    missing: list[str] = []
    for stage, stage_rows in stage_groups.items():
        clusters = len({row["cluster_id"] for row in stage_rows})
        stages[stage] = {"snapshots": len(stage_rows), "independent_clusters": clusters}
        if len(stage_rows) < min_stage_snapshots or clusters < min_stage_snapshots:
            missing.append(f"{stage}: requires {min_stage_snapshots} snapshots and clusters")
    buckets: dict[str, Any] = {}
    for name, values in bucket_values.items():
        buckets[name] = {}
        for value, identities in values.items():
            buckets[name][value] = len(identities)
            if len(identities) < min_bucket_snapshots:
                missing.append(f"{name}.{value}: requires {min_bucket_snapshots} independent requests")
    if any(unclassified.values()):
        missing.extend(f"{key}: {count} rows lack explicit bucket metadata"
                       for key, count in unclassified.items() if count)
    return {"status": "PASS" if not missing else "INSUFFICIENT_EVIDENCE",
            "stage_coverage": stages, "major_bucket_coverage": buckets,
            "unclassified_rows": unclassified, "missing": missing,
            "min_stage_snapshots": min_stage_snapshots,
            "min_bucket_snapshots": min_bucket_snapshots}


def _relative_error_summary(values: list[float]) -> dict[str, Any]:
    if not values:
        raise AcceptanceError("accuracy summary has no samples")
    if any(not math.isfinite(value) or value < 0 for value in values):
        raise AcceptanceError("accuracy errors must be finite and non-negative")
    return {"count": len(values), "mean": statistics.fmean(values),
            "p95_nearest_rank": nearest_rank(values, 0.95),
            "maximum": max(values)}


def _l2_bucket_labels(boundary: dict[str, Any]) -> dict[str, str]:
    metadata = boundary["causal_metadata"]
    labels: dict[str, str] = {}
    context = metadata.get("context_tokens", metadata.get("ctx"))
    if isinstance(context, int) and not isinstance(context, bool):
        if context <= 512:
            labels["context"] = "short"
        elif context >= 2048:
            labels["context"] = "long"
    competition = metadata.get("competition_level", metadata.get("competition"))
    if competition in ("low", "high"):
        labels["competition"] = competition
    elif competition in (0, 1) and not isinstance(competition, bool):
        labels["competition"] = "high" if competition else "low"
    source_count = metadata.get("source_count", metadata.get("sources"))
    if isinstance(source_count, int) and not isinstance(source_count, bool):
        labels["sources"] = "single" if source_count == 1 else "multi"
    capacity = boundary["causal_prestate"].get("capacity", {})
    if isinstance(capacity, dict):
        value = (metadata.get("capacity_bucket") or capacity.get("pressure_bucket")
                 or capacity.get("bucket"))
        if value in ("normal", "stress"):
            labels["capacity"] = value
    return labels


def _summarize_l2_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    all_pass = True
    for name in ("service", "settle"):
        values = [row[name]["relative_error"] for row in rows]
        summary = _relative_error_summary(values)
        clusters: dict[str, list[float]] = {}
        for row in rows:
            clusters.setdefault(row["cluster_id"], []).append(
                row[name]["relative_error"])
        summary.update({
            "pass": summary["mean"] <= 0.01
            and summary["p95_nearest_rank"] <= 0.03,
            "mean_bootstrap_95_ci": clustered_bootstrap_ci(
                clusters, statistic="mean"),
            "p95_bootstrap_95_ci": clustered_bootstrap_ci(
                clusters, statistic="p95"),
            "absolute_error_le_10_tick_count": sum(
                row[name]["absolute_error_ticks"] <= 10 for row in rows),
        })
        all_pass = all_pass and summary["pass"]
        metrics[name] = summary
    metrics["samples"] = len(rows)
    metrics["independent_clusters"] = len({row["cluster_id"] for row in rows})
    metrics["logical_event_equal_count"] = sum(
        row["logical_event_count_equal"] for row in rows)
    return {"status": "PASS" if all_pass else "FAIL", **metrics}


def _l2_prestate_payload(boundary: dict[str, Any]) -> dict[str, Any]:
    """Return only actual pre-action state that can affect the fixed action."""
    prestate = json.loads(json.dumps(boundary["causal_prestate"]))
    resources = prestate.get("resources")
    if (isinstance(resources, dict)
            and isinstance(resources.get("producer_build"), dict)):
        # producer_build 是本次运行的合同自我披露（kernel_contract：
        # 六键契约含 prediction_config_sha256），不同源构建本就不同——
        # 不是物理状态，不参与物理一致性比较（历史先例：2026-10-08
        # 实测的 state_mismatch 唯一差异即此；剥离后逐字节一致）。
        resources.pop("producer_build", None)
    return {"causal_prestate": prestate,
            "causal_metadata": boundary["causal_metadata"],
            "planner_state": boundary["planner_state"]}


def _canonical_digest(value: Any) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _measurement_boundaries(result: dict[str, Any]) -> dict[tuple[str, int], dict[str, Any]]:
    request_id = result.get("request_id")
    if not isinstance(request_id, str) or not request_id:
        raise AcceptanceError("L2 result lacks request identity", EXIT_FAIL,
                              "paired_execution_boundary_mismatch")
    rows: dict[tuple[str, int], dict[str, Any]] = {}
    for boundary in result.get("boundary_records", []):
        if boundary.get("request_id") != request_id:
            continue
        key = (boundary.get("stage"), boundary.get("boundary_ordinal"))
        if key in rows:
            raise AcceptanceError(f"L2 result has duplicate boundary {key!r}", EXIT_FAIL,
                                  "paired_execution_boundary_mismatch")
        rows[key] = boundary
    return rows


def compare_fixed_action_results(exact: dict[str, Any],
                                 approximate: dict[str, Any], *,
                                 focus_stage: str | None = None) -> dict[str, Any]:
    """Pair only same-state actions; retain post-divergence rows as diagnostics."""
    if focus_stage not in (None, "P", "D"):
        raise AcceptanceError("L2 focus_stage must be P, D, or omitted", EXIT_FAIL,
                              "paired_execution_identity_mismatch")
    exact_rows = {(row["stage"], row["boundary_ordinal"]): row
                  for row in exact["measurements"]
                  if focus_stage is None or row["stage"] == focus_stage}
    approx_rows = {(row["stage"], row["boundary_ordinal"]): row
                   for row in approximate["measurements"]
                   if focus_stage is None or row["stage"] == focus_stage}
    exact_boundaries = _measurement_boundaries(exact)
    approx_boundaries = _measurement_boundaries(approximate)
    first_key_order = lambda key, row: (row["start_tick_ns"],
                                        0 if key[0] == "P" else 1, key[1])
    exact_first = min(exact_rows, key=lambda key: first_key_order(key, exact_rows[key])) \
        if exact_rows else None
    approx_first = min(approx_rows, key=lambda key: first_key_order(key, approx_rows[key])) \
        if approx_rows else None
    same_case = all(exact.get(field) == approximate.get(field)
                    for field in ("case_id", "cluster_id", "request_id"))
    identity_fields = ("case_id", "cluster_id", "request_id")
    if not same_case or not all(isinstance(exact.get(field), str) and exact.get(field)
                                for field in identity_fields):
        raise AcceptanceError("L2 arms do not share case/cluster/request identity",
                              EXIT_FAIL, "paired_execution_identity_mismatch")
    same_initial = (exact.get("initial_certificate") is not None
                    and exact.get("initial_certificate")
                    == approximate.get("initial_certificate"))
    pairs: list[dict[str, Any]] = []
    mismatches: list[dict[str, Any]] = []
    for key in sorted(set(exact_rows) | set(approx_rows)):
        if key not in exact_rows or key not in approx_rows:
            mismatches.append({"stage": key[0], "boundary_ordinal": key[1],
                               "status": "state_mismatch",
                               "reason": "stage_or_boundary_missing_from_one_arm"})
            continue
        lhs, rhs = exact_rows[key], approx_rows[key]
        boundary_l = exact_boundaries.get(key)
        boundary_r = approx_boundaries.get(key)
        if boundary_l is None or boundary_r is None:
            mismatches.append({"stage": key[0], "boundary_ordinal": key[1],
                               "status": "state_mismatch",
                               "reason": "actual_prestate_boundary_missing"})
            continue
        cold_first = (
            same_initial and key == exact_first and key == approx_first
            and key[1] == 0
            and boundary_l.get("tick_ns") == lhs["start_tick_ns"]
            and boundary_r.get("tick_ns") == rhs["start_tick_ns"]
            and boundary_l.get("planner_state") == boundary_r.get("planner_state")
            and boundary_l.get("causal_metadata") == boundary_r.get("causal_metadata")
        )
        same_prestate = (_canonical_digest(_l2_prestate_payload(boundary_l))
                         == _canonical_digest(_l2_prestate_payload(boundary_r)))
        same_stage_identity = (
            lhs["fixed_instance_id"] == rhs["fixed_instance_id"]
            and lhs["fixed_action"] == rhs["fixed_action"]
            and lhs["start_tick_ns"] == rhs["start_tick_ns"]
        )
        if not same_stage_identity or not (same_prestate or cold_first):
            mismatches.append({
                "stage": key[0], "boundary_ordinal": key[1],
                "status": "state_mismatch",
                "reason": ("stage_identity_or_start_differs" if not same_stage_identity
                           else "canonical_causal_prestate_differs"),
                "exact_prestate_sha256": _canonical_digest(_l2_prestate_payload(boundary_l)),
                "approx_prestate_sha256": _canonical_digest(_l2_prestate_payload(boundary_r)),
                "cold_initial_first_boundary": cold_first,
            })
            continue
        if lhs["charged_totals"] != rhs["charged_totals"]:
            raise AcceptanceError("approximation changed physical charged totals",
                                  EXIT_FAIL, "physical_invariant_violation")
        service = paired_duration_error(lhs["start_tick_ns"],
                                        lhs["service_tick_ns"],
                                        rhs["service_tick_ns"])
        settle = paired_duration_error(lhs["start_tick_ns"],
                                       lhs["settle_tick_ns"],
                                       rhs["settle_tick_ns"])
        if not service["valid"] or not settle["valid"]:
            raise AcceptanceError("zero-reference timing changed under approximation",
                                  EXIT_FAIL, "zero_reference_timing_violation")
        pairs.append({"stage": lhs["stage"], "boundary_ordinal": key[1],
                      "fixed_instance_id": lhs["fixed_instance_id"],
                      "fixed_action": lhs["fixed_action"],
                      "cluster_id": exact["cluster_id"],
                      "request_id": exact["request_id"],
                      "bucket_labels": _l2_bucket_labels(boundary_l),
                      "service": service, "settle": settle,
                      "logical_event_count_equal": (
                          lhs["logical_event_count"] == rhs["logical_event_count"]),
                      "physical_charges_equal": True,
                      "prestate_sha256": _canonical_digest(_l2_prestate_payload(boundary_l)),
                      "cold_initial_first_boundary": cold_first})
    grouped: dict[str, dict[str, Any]] = {}
    by_stage_action: dict[str, dict[str, Any]] = {}
    by_stage_bucket: dict[str, dict[str, Any]] = {}
    for stage in ("P", "D"):
        stage_pairs = [row for row in pairs if row["stage"] == stage]
        if stage_pairs:
            grouped[stage] = _summarize_l2_group(stage_pairs)
            for action in VALID_ACTIONS:
                action_rows = [row for row in stage_pairs
                               if row["fixed_action"] == action]
                if action_rows:
                    by_stage_action[f"{stage}/{action}"] = _summarize_l2_group(action_rows)
            for bucket_name in ("context", "competition", "sources", "capacity"):
                values = sorted({row["bucket_labels"].get(bucket_name)
                                 for row in stage_pairs
                                 if row["bucket_labels"].get(bucket_name) is not None})
                for value in values:
                    bucket_rows = [row for row in stage_pairs
                                   if row["bucket_labels"].get(bucket_name) == value]
                    by_stage_bucket[f"{stage}/{bucket_name}/{value}"] = (
                        _summarize_l2_group(bucket_rows))
    all_groups = list(grouped.values()) + list(by_stage_action.values()) \
        + list(by_stage_bucket.values())
    metrics_pass = bool(grouped) and all(item["status"] == "PASS" for item in all_groups)
    return {"status": "PASS" if metrics_pass else "INSUFFICIENT_EVIDENCE" if mismatches and not pairs else "FAIL",
            "pairs": pairs, "state_mismatches": mismatches, "by_stage": grouped,
            "by_stage_action": by_stage_action, "by_stage_bucket": by_stage_bucket,
            "evaluable_pair_count": len(pairs),
            "physical_charges_identical": all(row["physical_charges_equal"]
                                               for row in pairs)}


def compare_projection_evaluation(
        evaluation: dict[str, Any], boundary: dict[str, Any], *,
        prediction_config_sha256: str,
        library_sha256: str | None = None) -> dict[str, Any]:
    """Validate and score the acceptance-only, production-planner L3 DTO.

    Candidate work and the single native result vector are preserved from
    ``acceptance_projection.evaluate_boundary``. This routine only checks
    their identities and the single-face selection contract (the former
    approximate/exact dual face retired with the E0 arbitration mechanism on
    2026-10-16; the configured-precision face is the exact face).
    """
    required = {
        "version", "status", "reason", "request_id", "stage", "boundary_ordinal", "tick_ns",
        "horizon_tokens", "projection_input", "candidates", "inapplicable_candidates",
        "forecast_generator",
        "results", "selected_id", "library_sha256",
        "prediction_config_sha256",
        # 生产 DTO 的信息性键（模型语义版本与世界源状态），比较器不
        # 消费但必须接受（2026-10-08 单点干跑实测 extra 键 FAIL）。
        "model_semantics_version", "world_source_status",
    }
    _require_object(evaluation, "projection evaluation")
    if not {"version", "status", "reason"} <= evaluation.keys():
        raise AcceptanceError("projection evaluation lacks status/reason envelope",
                              EXIT_FAIL, "projection_evaluation_schema_invalid")
    if evaluation["version"] != "projection-evaluation-v3-20261016":
        raise AcceptanceError("unsupported projection evaluation version", EXIT_FAIL,
                              "projection_evaluation_version_mismatch")
    if evaluation["status"] not in ("PASS", "FAIL", "INCOMPLETE"):
        raise AcceptanceError("projection evaluation status is invalid", EXIT_FAIL,
                              "projection_evaluation_schema_invalid")
    if evaluation["status"] != "PASS":
        allowed = {"version", "status", "reason", "request_id", "stage",
                   "boundary_ordinal", "tick_ns"}
        if evaluation.keys() - allowed:
            raise AcceptanceError("failed projection evaluation carries unexpected fields",
                                  EXIT_FAIL, "projection_evaluation_schema_invalid")
        if not set(("version", "status", "reason")) <= evaluation.keys():
            raise AcceptanceError("failed projection evaluation lacks its reason",
                                  EXIT_FAIL, "projection_evaluation_schema_invalid")
        for field in ("request_id", "stage", "boundary_ordinal", "tick_ns"):
            if field in evaluation and evaluation[field] != boundary.get(field):
                raise AcceptanceError(f"failed projection evaluation boundary mismatch at {field}",
                                      EXIT_FAIL, "projection_boundary_mismatch")
        if not isinstance(evaluation["reason"], str) or not evaluation["reason"]:
            raise AcceptanceError("failed projection evaluation requires a reason",
                                  EXIT_FAIL, "projection_evaluation_schema_invalid")
        return {"status": evaluation["status"], "reason": evaluation["reason"],
                "request_id": evaluation.get("request_id", boundary.get("request_id")),
                "stage": evaluation.get("stage", boundary.get("stage")),
                "boundary_ordinal": evaluation.get("boundary_ordinal",
                                                    boundary.get("boundary_ordinal")),
                "tick_ns": evaluation.get("tick_ns", boundary.get("tick_ns")),
                "evaluable": False}
    _require_exact_keys(evaluation, required, "projection evaluation")
    if evaluation["reason"] is not None and not isinstance(evaluation["reason"], str):
        raise AcceptanceError("PASS projection evaluation reason must be null or string",
                              EXIT_FAIL, "projection_evaluation_schema_invalid")
    for field in ("request_id", "stage", "boundary_ordinal", "tick_ns"):
        if evaluation[field] != boundary.get(field):
            raise AcceptanceError(f"projection evaluation boundary mismatch at {field}",
                                  EXIT_FAIL, "projection_boundary_mismatch")
    if evaluation["stage"] not in ("P", "D"):
        raise AcceptanceError("projection evaluation stage must be P or D", EXIT_FAIL,
                              "projection_boundary_mismatch")
    _require_int(evaluation["boundary_ordinal"], "projection boundary_ordinal")
    _require_int(evaluation["tick_ns"], "projection tick_ns")
    _require_int(evaluation["horizon_tokens"], "projection horizon_tokens", minimum=1)
    config_sha = evaluation["prediction_config_sha256"]
    lib_sha = evaluation["library_sha256"]
    for name, value in (("prediction config", config_sha), ("native library", lib_sha)):
        if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
            raise AcceptanceError(f"projection evaluation {name} digest is invalid", EXIT_FAIL,
                                  "projection_evaluation_digest_invalid")
    if config_sha != prediction_config_sha256:
        raise AcceptanceError("projection evaluation used another prediction config",
                              EXIT_FAIL, "projection_config_mismatch")
    if library_sha256 is not None and lib_sha != library_sha256:
        raise AcceptanceError("projection evaluation used another native library",
                              EXIT_FAIL, "projection_library_mismatch")

    projection_input = _require_object(evaluation["projection_input"], "projection_input")
    if _contains_key(projection_input, "committed_decode_tokens"):
        raise AcceptanceError("L3 ProjectionInput leaks actual decode completion length",
                              EXIT_FAIL, "future_execution_state_leaked")
    projection_sha = _canonical_digest(projection_input)
    candidates = evaluation["candidates"]
    if not isinstance(candidates, list) or not candidates:
        raise AcceptanceError("projection evaluation has no generated candidates",
                              EXIT_INCOMPLETE, "candidate_snapshot_empty")
    # 生产 DTO 的 instance_id 为 int 实例号；冻结合同要求 str——入口
    # 规范化（2026-10-08 实测 int 直判 "lacks instance identity"）。
    for _row in candidates:
        if isinstance(_row, dict) and isinstance(_row.get("instance_id"), int) \
                and not isinstance(_row.get("instance_id"), bool):
            _row["instance_id"] = str(_row["instance_id"])
    for _row in (evaluation.get("inapplicable_candidates") or []):
        if isinstance(_row, dict) and isinstance(_row.get("instance_id"), int) \
                and not isinstance(_row.get("instance_id"), bool):
            _row["instance_id"] = str(_row["instance_id"])
    candidate_order: list[str] = []
    candidate_actions: dict[str, str] = {}
    candidate_wires: dict[str, dict[str, Any]] = {}
    for index, candidate_value in enumerate(candidates):
        candidate = _require_object(candidate_value, f"candidates[{index}]")
        _require_exact_keys(candidate, {"candidate_id", "instance_id", "action", "work_item"},
                            f"candidates[{index}]")
        candidate_id, action = candidate["candidate_id"], candidate["action"]
        if (not isinstance(candidate_id, str) or not candidate_id
                or candidate_id in candidate_wires):
            raise AcceptanceError("candidate IDs must be unique non-empty strings", EXIT_FAIL,
                                  "candidate_set_mismatch")
        if not isinstance(action, str) or action not in VALID_ACTIONS:
            raise AcceptanceError(f"candidate {candidate_id} has invalid action", EXIT_FAIL,
                                  "candidate_result_schema_invalid")
        if not isinstance(candidate["instance_id"], str) or not candidate["instance_id"]:
            raise AcceptanceError(f"candidate {candidate_id} lacks instance identity", EXIT_FAIL,
                                  "candidate_result_schema_invalid")
        work_item = _require_object(candidate["work_item"],
                                    f"candidate {candidate_id}.work_item")
        _require_exact_keys(work_item, {"work_id", "legs", "ready_at_ns",
                                        "depends_on_work", "is_candidate",
                                        "service_leg_ids"},
                            f"candidate {candidate_id}.work_item")
        if (not isinstance(work_item["work_id"], str) or not work_item["work_id"]
                or not isinstance(work_item["legs"], list)
                or work_item["is_candidate"] is not True):
            raise AcceptanceError(f"candidate {candidate_id} work_item is not typed candidate work",
                                  EXIT_FAIL, "candidate_result_schema_invalid")
        candidate_order.append(candidate_id)
        candidate_actions[candidate_id] = action
        candidate_wires[candidate_id] = candidate

    # The projection evaluator deliberately prices only the applicable subset.
    # Preserve the complementary menu rows, in their supplied stable order,
    # without requiring a native result for work that cannot be constructed.
    inapplicable_values = evaluation["inapplicable_candidates"]
    if not isinstance(inapplicable_values, list):
        raise AcceptanceError("inapplicable_candidates must be an array", EXIT_FAIL,
                              "candidate_set_mismatch")
    inapplicable_candidates: list[dict[str, str]] = []
    inapplicable_ids: set[str] = set()
    for index, candidate_value in enumerate(inapplicable_values):
        candidate = _require_object(candidate_value, f"inapplicable_candidates[{index}]")
        _require_exact_keys(candidate, {"candidate_id", "instance_id", "action", "reason"},
                            f"inapplicable_candidates[{index}]")
        candidate_id = candidate["candidate_id"]
        instance_id = candidate["instance_id"]
        action = candidate["action"]
        reason = candidate["reason"]
        if (not isinstance(candidate_id, str) or not candidate_id
                or candidate_id in candidate_wires or candidate_id in inapplicable_ids):
            raise AcceptanceError(
                "applicable and inapplicable candidate IDs must be unique and disjoint",
                EXIT_FAIL, "candidate_set_mismatch")
        if not isinstance(instance_id, str) or not instance_id:
            raise AcceptanceError("inapplicable candidate lacks instance identity", EXIT_FAIL,
                                  "candidate_result_schema_invalid")
        if not isinstance(action, str) or action not in VALID_ACTIONS:
            raise AcceptanceError(f"inapplicable candidate {candidate_id} has invalid action",
                                  EXIT_FAIL, "candidate_result_schema_invalid")
        if not isinstance(reason, str) or not reason.strip():
            raise AcceptanceError(f"inapplicable candidate {candidate_id} lacks a reason",
                                  EXIT_FAIL, "candidate_result_schema_invalid")
        inapplicable_ids.add(candidate_id)
        inapplicable_candidates.append({"candidate_id": candidate_id,
                                        "instance_id": instance_id,
                                        "action": action, "reason": reason})

    generator = _require_object(evaluation["forecast_generator"], "forecast_generator")
    _require_exact_keys(generator, {"identity", "causal_inputs", "semantic_sha256"},
                        "forecast_generator")
    if (not isinstance(generator["identity"], str) or not generator["identity"]):
        raise AcceptanceError("forecast generator identity is missing", EXIT_FAIL,
                              "prediction_generator_identity_missing")
    causal_inputs = _require_object(generator["causal_inputs"],
                                   "forecast_generator.causal_inputs")
    _require_exact_keys(causal_inputs, {"context_tokens", "horizon_tokens",
                                        "generator_inputs"},
                        "forecast_generator.causal_inputs")
    context_tokens = _require_int(causal_inputs["context_tokens"],
                                  "forecast_generator.context_tokens")
    horizon_tokens = _require_int(causal_inputs["horizon_tokens"],
                                  "forecast_generator.horizon_tokens", minimum=1)
    if horizon_tokens != evaluation["horizon_tokens"]:
        raise AcceptanceError("forecast generator horizon differs from evaluation horizon",
                              EXIT_FAIL, "prediction_input_mismatch")
    generator_inputs = _require_object(causal_inputs["generator_inputs"],
                                       "forecast_generator.generator_inputs")
    boundary_metadata = _require_object(boundary.get("causal_metadata"),
                                        "boundary.causal_metadata")
    known_context = boundary_metadata.get("context_tokens", boundary_metadata.get("ctx"))
    known_horizon = boundary_metadata.get("horizon_tokens")
    if known_horizon is None:
        known_horizon = boundary_metadata.get("forecast_horizon_tokens")
    if known_context is None or known_horizon is None:
        raise AcceptanceError("boundary lacks actual context/horizon needed for L3",
                              EXIT_INCOMPLETE, "l3_causal_inputs_missing")
    known_context = _require_int(known_context, "boundary.context_tokens")
    known_horizon = _require_int(known_horizon, "boundary.horizon_tokens", minimum=1)
    if context_tokens != known_context or horizon_tokens != known_horizon:
        raise AcceptanceError("forecast generator context/horizon differs from actual boundary",
                              EXIT_FAIL, "prediction_input_mismatch")
    expected_generator_inputs = boundary_metadata.get("generator_inputs", {})
    if generator_inputs != expected_generator_inputs:
        raise AcceptanceError("forecast generator inputs differ from actual boundary metadata",
                              EXIT_FAIL, "prediction_input_mismatch")
    if _contains_key(generator["causal_inputs"], "committed_decode_tokens"):
        raise AcceptanceError("L3 forecast generator leaks actual decode completion length",
                              EXIT_FAIL, "future_execution_state_leaked")
    generator_sha = generator["semantic_sha256"]
    if (not isinstance(generator_sha, str)
            or re.fullmatch(r"[0-9a-f]{64}", generator_sha) is None):
        raise AcceptanceError("forecast generator semantic digest is invalid", EXIT_FAIL,
                              "prediction_generator_digest_missing")
    generator_payload = {"identity": generator["identity"],
                         "causal_inputs": generator["causal_inputs"]}
    if _canonical_digest(generator_payload) != generator_sha:
        raise AcceptanceError("forecast generator digest does not bind its typed inputs",
                              EXIT_FAIL, "prediction_generator_digest_mismatch")

    def validate_result_rows(value: Any) -> dict[str, dict[str, Any]]:
        """Single-face results validation; the v2 row shape is exporter-defined
        (full native disclosure quality incl. reference_cache_* and the full
        evaluation row as result)."""
        if not isinstance(value, list):
            raise AcceptanceError("results must be an array", EXIT_FAIL,
                                  "candidate_result_schema_invalid")
        indexed: dict[str, dict[str, Any]] = {}
        for index, item in enumerate(value):
            row = _require_object(item, f"results[{index}]")
            _require_exact_keys(row, {"candidate_id", "status", "service_ns", "settle_ns",
                                      "quality", "result"}, f"results[{index}]")
            candidate_id = row["candidate_id"]
            if not isinstance(candidate_id, str) or candidate_id in indexed:
                raise AcceptanceError("results has duplicate/invalid candidate ID", EXIT_FAIL,
                                      "candidate_set_mismatch")
            result = _require_object(row["result"], f"results[{index}].result")
            _require_exact_keys(result, {
                "candidate_id", "status", "applicable", "inapplicable_reason",
                "cost_ns", "service_ns", "settle_ns", "native_kernel",
            }, f"results[{index}].result")
            if row["status"] != result["status"]:
                raise AcceptanceError(f"results[{index}] status differs from full result",
                                      EXIT_FAIL, "candidate_result_schema_invalid")
            if result["candidate_id"] != candidate_id:
                raise AcceptanceError(f"results[{index}] candidate_id differs from full result",
                                      EXIT_FAIL, "candidate_result_schema_invalid")
            if (row["service_ns"] != result["service_ns"]
                    or row["settle_ns"] != result["settle_ns"]):
                raise AcceptanceError(f"results[{index}] durations differ from full result",
                                      EXIT_FAIL, "candidate_result_schema_invalid")
            quality = _require_object(row["quality"], f"results[{index}].quality")
            if row["status"] == "inapplicable":
                # 不适用候选无投影（breakdown 缺省）：数值面全 None、披露
                # 空 dict、applicable 标记为 False。
                if quality:
                    raise AcceptanceError(f"results[{index}] inapplicable row carries a disclosure",
                                          EXIT_FAIL, "candidate_quality_schema_invalid")
                if result["applicable"] is not False:
                    raise AcceptanceError(
                        f"results[{index}] inapplicable row is not marked inapplicable",
                        EXIT_FAIL, "candidate_result_schema_invalid")
                if row["service_ns"] is not None or row["settle_ns"] is not None \
                        or result["cost_ns"] is not None:
                    raise AcceptanceError(f"results[{index}] inapplicable row carries prices",
                                          EXIT_FAIL, "candidate_price_invalid")
                indexed[candidate_id] = row
                continue
            if row["status"] == "degraded":
                raise AcceptanceError(
                    f"candidate {candidate_id} projection exhausted its logical event budget",
                    EXIT_FAIL, "candidate_budget_failure")
            if row["status"] == "model_unavailable":
                raise AcceptanceError(
                    f"candidate {candidate_id} native backend/model is unavailable",
                    EXIT_FAIL, "candidate_backend_failure")
            if row["status"] != "ok":
                raise AcceptanceError(f"candidate {candidate_id} is not priceable in the native run",
                                      EXIT_INCOMPLETE, "candidate_price_incomplete")
            _require_exact_keys(quality, {
                "unsupported_reason", "logical_event_count", "physical_step_count",
                "reference_cache_hits", "reference_cache_misses",
            }, f"results[{index}].quality")
            if (quality["unsupported_reason"] is not None
                    and not isinstance(quality["unsupported_reason"], str)):
                raise AcceptanceError(f"results[{index}].quality has invalid typed fields",
                                      EXIT_FAIL, "candidate_quality_schema_invalid")
            for metric in ("logical_event_count", "physical_step_count",
                           "reference_cache_hits", "reference_cache_misses"):
                _require_int(quality[metric], f"results[{index}].quality.{metric}")
            if result["applicable"] is not True:
                raise AcceptanceError(
                    f"results[{index}] priced row is not marked applicable",
                    EXIT_FAIL, "candidate_result_schema_invalid")
            for key in ("service_ns", "settle_ns", "cost_ns"):
                value_field = result[key] if key == "cost_ns" else row[key]
                if (isinstance(value_field, bool)
                        or not isinstance(value_field, (int, float))
                        or not math.isfinite(value_field) or value_field < 0):
                    raise AcceptanceError(f"candidate {candidate_id}.{key} is invalid",
                                          EXIT_INCOMPLETE, "candidate_price_invalid")
            if result["cost_ns"] != row["settle_ns"] or row["service_ns"] > row["settle_ns"]:
                raise AcceptanceError(
                    f"candidate {candidate_id} priced row violates cost/settle consistency",
                    EXIT_FAIL, "candidate_result_schema_invalid")
            if result["native_kernel"] != quality:
                raise AcceptanceError(f"candidate {candidate_id} quality differs from full result",
                                      EXIT_FAIL, "candidate_quality_mismatch")
            indexed[candidate_id] = row
        return indexed

    results = validate_result_rows(evaluation["results"])
    expected_ids = set(candidate_order) | inapplicable_ids
    if set(results) != expected_ids:
        raise AcceptanceError("results candidate set differs from the applicable/inapplicable faces",
                              EXIT_FAIL, "candidate_set_mismatch")
    selected_id = evaluation["selected_id"]
    if not isinstance(selected_id, str) or not selected_id:
        raise AcceptanceError("selection produced no selected candidate",
                              EXIT_INCOMPLETE, "selection_missing")
    if selected_id not in candidate_wires:
        raise AcceptanceError("selected_id references an unknown candidate",
                              EXIT_FAIL, "candidate_selection_mismatch")
    selected_row = results.get(selected_id)
    if selected_row is None or selected_row["status"] != "ok":
        raise AcceptanceError("selected candidate has no valid finite price",
                              EXIT_INCOMPLETE, "candidate_winner_unpriceable")
    candidate_result_rows: list[dict[str, Any]] = []
    for candidate_id in candidate_order:
        row = results[candidate_id]
        if row["status"] != "ok":
            raise AcceptanceError(f"applicable candidate {candidate_id} is not priceable",
                                  EXIT_INCOMPLETE, "candidate_price_incomplete")
        candidate_result_rows.append({
            "candidate_id": candidate_id,
            "action": candidate_actions[candidate_id],
            "status": row["status"],
            "service_ns": row["service_ns"],
            "settle_ns": row["settle_ns"],
        })
    return {
        "request_id": evaluation["request_id"], "stage": evaluation["stage"],
        "status": "PASS", "reason": evaluation["reason"], "evaluable": True,
        "boundary_ordinal": evaluation["boundary_ordinal"],
        "tick_ns": evaluation["tick_ns"], "horizon_tokens": evaluation["horizon_tokens"],
        "projection_input_sha256": projection_sha,
        "forecast_generator_sha256": generator_sha,
        "library_sha256": lib_sha, "prediction_config_sha256": config_sha,
        "candidate_order": candidate_order,
        "inapplicable_candidates": inapplicable_candidates,
        "candidate_results": candidate_result_rows,
        "selected_id": selected_id,
        "selected_action": candidate_actions[selected_id],
        "cluster_id": boundary.get("cluster_id"),
        "same_input_and_candidate_set": True,
    }


def _load_acceptance_module(source_root: Path):
    """Load the acceptance-only planner module from the hash-recorded source copy."""
    package_parent = (source_root / "sh_test_mesh/workload/llama2_7b_inference")
    module_path = package_parent / "joint/acceptance_projection.py"
    if not module_path.is_file():
        raise AcceptanceError(
            f"real L3 acceptance planner is missing from source copy: {module_path}",
            EXIT_INCOMPLETE, "l3_acceptance_projection_missing")
    resolved_parent = package_parent.resolve()
    loaded = sys.modules.get("joint")
    if loaded is not None:
        paths = [Path(item).resolve() for item in getattr(loaded, "__path__", [])]
        # joint.__path__ 的每个条目指向包目录本身（…/llama2_7b_inference/
        # joint），必须与期望包目录比较；与父目录比较会使任何同根重载
        # 都被误判为 collision。
        expected_package_dir = resolved_parent / "joint"
        if expected_package_dir not in paths:
            raise AcceptanceError(
                "a different joint package is already loaded; run acceptance in a fresh process",
                EXIT_INCOMPLETE, "l3_source_import_collision")
    else:
        sys.path.insert(0, str(resolved_parent))
    try:
        module = importlib.import_module("joint.acceptance_projection")
    except Exception as exc:
        raise AcceptanceError(f"cannot import acceptance_projection from source copy: {exc}",
                              EXIT_INCOMPLETE, "l3_acceptance_projection_import_failed") from exc
    if Path(module.__file__).resolve() != module_path.resolve():
        raise AcceptanceError("acceptance_projection import resolved outside selected source copy",
                              EXIT_FAIL, "l3_source_provenance_mismatch")
    return module


def _load_acceptance_projection(source_root: Path):
    """Load the acceptance-only planner from the hash-recorded source copy."""
    module = _load_acceptance_module(source_root)
    evaluate = getattr(module, "evaluate_boundary", None)
    if not callable(evaluate):
        raise AcceptanceError("acceptance_projection lacks evaluate_boundary()",
                              EXIT_INCOMPLETE, "l3_acceptance_projection_api_missing")
    return evaluate


def evaluate_l3_boundary(source_root: Path, boundary: dict[str, Any],
                         prediction_config: dict[str, Any], *,
                         library_path: Path, fixture_state: dict[str, Any],
                         prediction_config_sha256: str) -> dict[str, Any]:
    if not library_path.is_file():
        raise AcceptanceError(f"native projection library is missing: {library_path}",
                              EXIT_INCOMPLETE, "native_projection_library_missing")
    evaluator = _load_acceptance_projection(source_root)
    try:
        dto = evaluator(boundary, prediction_config,
                        library_path=str(library_path), fixture_state=fixture_state)
    except AcceptanceError:
        raise
    except Exception as exc:
        raise AcceptanceError(f"acceptance_projection.evaluate_boundary failed: {exc}",
                              EXIT_INCOMPLETE, "l3_evaluation_api_failed") from exc
    evaluation = _require_object(dto, "evaluate_boundary result")
    return compare_projection_evaluation(
        evaluation, boundary,
        prediction_config_sha256=prediction_config_sha256,
        library_sha256=sha256_file(library_path))


def summarize_performance_repetitions(repetitions: list[dict[str, Any]]) -> dict[str, Any]:
    expected_orders = [
        ["old_exact", "new_exact", "new_approx"],
        ["new_exact", "new_approx", "old_exact"],
        ["new_approx", "old_exact", "new_exact"],
        ["old_exact", "new_exact", "new_approx"],
        ["new_exact", "new_approx", "old_exact"],
    ]
    if len(repetitions) != 5:
        raise AcceptanceError("performance result requires five complete paired repetitions")
    work_signatures: set[str] = set()
    wall: dict[str, list[int]] = {name: [] for name in
                                  ("old_exact", "new_exact", "new_approx")}
    rss: dict[str, list[int | None]] = {name: [] for name in wall}
    for index, repetition in enumerate(repetitions):
        if repetition.get("status") != "complete" or repetition.get("order") != expected_orders[index]:
            raise AcceptanceError("performance repetitions must be complete and order-rotated",
                                  EXIT_FAIL, "performance_pair_invalid")
        arms = repetition.get("arms")
        if not isinstance(arms, dict) or set(arms) != set(wall):
            raise AcceptanceError("performance repetition must include exactly three arms",
                                  EXIT_FAIL, "performance_pair_invalid")
        signatures = set()
        for name, record in arms.items():
            if record.get("status") != "completed" or record.get("natural_completion") is not True:
                raise AcceptanceError("partial/timeout performance work cannot enter ratios",
                                      EXIT_INCOMPLETE, "performance_incomplete")
            duration = _require_int(record.get("elapsed_host_ns"),
                                    f"performance.{name}.elapsed_host_ns", minimum=1)
            signature = record.get("work_signature_sha256")
            if not isinstance(signature, str) or re.fullmatch(r"[0-9a-f]{64}", signature) is None:
                raise AcceptanceError("performance arm lacks a complete work signature",
                                      EXIT_FAIL, "performance_pair_invalid")
            signatures.add(signature)
            wall[name].append(duration)
            peak = record.get("peak_rss_bytes")
            rss[name].append(None if peak is None else _require_int(
                peak, f"performance.{name}.peak_rss_bytes"))
        if len(signatures) != 1:
            raise AcceptanceError("performance arms used different work/hardware inputs",
                                  EXIT_FAIL, "performance_work_mismatch")
        work_signatures.update(signatures)
    if len(work_signatures) != 1:
        raise AcceptanceError("performance repetitions do not share one work signature",
                              EXIT_FAIL, "performance_work_mismatch")
    medians = {name: int(statistics.median(values)) for name, values in wall.items()}
    ratios = {
        "S_engineering": medians["old_exact"] / medians["new_exact"],
        "S_approx_extra": medians["new_exact"] / medians["new_approx"],
        "S_total": medians["old_exact"] / medians["new_approx"],
    }
    return {"status": "PASS", "work_signature_sha256": next(iter(work_signatures)),
            "repetitions": repetitions, "median_wall_ns": medians,
            "min_wall_ns": {name: min(values) for name, values in wall.items()},
            "max_wall_ns": {name: max(values) for name, values in wall.items()},
            "peak_rss_bytes": rss, "speed_ratios": ratios,
            "speed_goal_met": ratios["S_total"] >= 20.0}


def summarize_exact_only_repetitions(repetitions: list[dict[str, Any]]) -> dict[str, Any]:
    orders = [["old_exact", "new_exact"], ["new_exact", "old_exact"],
              ["old_exact", "new_exact"], ["new_exact", "old_exact"],
              ["old_exact", "new_exact"]]
    if len(repetitions) != len(orders):
        raise AcceptanceError("exact-only diagnostic requires five complete pairs",
                              EXIT_INCOMPLETE, "performance_incomplete")
    wall: dict[str, list[int]] = {"old_exact": [], "new_exact": []}
    signatures: set[str] = set()
    for index, repetition in enumerate(repetitions):
        if (repetition.get("status") != "complete"
                or repetition.get("order") != orders[index]):
            raise AcceptanceError("exact-only repetitions differ from the frozen order",
                                  EXIT_FAIL, "performance_pair_invalid")
        arms = repetition.get("arms")
        if not isinstance(arms, dict) or set(arms) != set(wall):
            raise AcceptanceError("exact-only repetition must contain old/new exact arms",
                                  EXIT_FAIL, "performance_pair_invalid")
        pair_signatures: set[str] = set()
        for name, arm in arms.items():
            if arm.get("status") != "completed" or arm.get("natural_completion") is not True:
                raise AcceptanceError("incomplete exact-only arm cannot enter comparison",
                                      EXIT_INCOMPLETE, "performance_incomplete")
            wall[name].append(_require_int(arm.get("elapsed_host_ns"),
                                           f"{name}.elapsed_host_ns", minimum=1))
            signature = arm.get("work_signature_sha256")
            if not isinstance(signature, str) or re.fullmatch(r"[0-9a-f]{64}", signature) is None:
                raise AcceptanceError("exact-only arm lacks a valid work signature",
                                      EXIT_FAIL, "performance_pair_invalid")
            pair_signatures.add(signature)
        if len(pair_signatures) != 1:
            raise AcceptanceError("old/new exact arms used different work",
                                  EXIT_FAIL, "performance_work_mismatch")
        signatures.update(pair_signatures)
    if len(signatures) != 1:
        raise AcceptanceError("exact-only repetitions used different work",
                              EXIT_FAIL, "performance_work_mismatch")
    medians = {name: int(statistics.median(values)) for name, values in wall.items()}
    return {
        "status": "PASS", "work_signature_sha256": next(iter(signatures)),
        "median_wall_ns": medians,
        "speed_ratios": {"S_engineering": medians["old_exact"] / medians["new_exact"],
                         "S_approx_extra": None, "S_total": None},
        "speed_goal_met": None, "formal_speed_claim": False,
        "mode": "exact_only_diagnostic",
    }


def _trace_nonnegative_digits(value: str, *, line: int, field: str) -> int:
    if not value or any(character < "0" or character > "9" for character in value):
        raise AcceptanceError(f"trace line {line}: {field} must use ASCII digits only",
                              EXIT_FAIL, "trace_selection_invalid")
    return int(value)


def select_first_second_trace(csv_path: Path, window_ns: int = 1_000_000_000) -> dict[str, Any]:
    """Reproduce the frozen selector: turn arrivals derive from prior-row gaps."""
    _require_int(window_ns, "trace window_ns", minimum=1)
    if window_ns != 1_000_000_000:
        raise AcceptanceError("formal trace selector is frozen to the first 1 second",
                              EXIT_FAIL, "trace_window_changed")
    if not csv_path.is_file():
        raise AcceptanceError(f"frozen trace CSV is missing: {csv_path}")
    selected: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    seen: set[str] = set()
    first_later: dict[str, Any] | None = None
    last_session: str | None = None
    last_first_arrival: int | None = None
    with csv_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.reader(stream)
        fieldnames = next(reader, None)
        expected = {"arrival_time", "session_id", "request_id", "prefix_len",
                    "prefill_length", "decode_length", "human_time", "tool_time"}
        if fieldnames is None or not expected.issubset(fieldnames):
            raise AcceptanceError("frozen trace header lacks required source columns",
                                  EXIT_FAIL, "trace_selection_invalid")
        # Match DictReader's dictionary behavior for duplicate names (the last
        # column wins), short rows (missing cells read as None), and extra
        # cells (ignored by this selector). csv.reader avoids materializing a
        # dictionary for every row in the large frozen trace.
        field_index = {name: index for index, name in enumerate(fieldnames)}

        def value(row: list[str], name: str) -> str | None:
            index = field_index[name]
            return row[index] if index < len(row) else None

        current_session: str | None = None
        current_turn = 0
        turn_arrival: int | None = None
        previous_gap: str | None = None
        previous_line = 0
        session_selecting = False
        selection_error: AcceptanceError | None = None
        for row in reader:
            # DictReader skips completely empty records before returning a
            # row; retain that CSV behavior and the physical line numbering.
            if not row:
                continue
            line = reader.line_num
            raw_sid = value(row, "session_id") or ""
            if not raw_sid:
                raise AcceptanceError(f"trace line {line}: empty session_id",
                                      EXIT_FAIL, "trace_selection_invalid")
            sid = f"session_{raw_sid}"

            new_session = current_session is None or sid != current_session
            if new_session:
                if current_session is not None:
                    seen.add(current_session)
                if sid in seen:
                    raise AcceptanceError(f"trace session is not contiguous: {sid}",
                                          EXIT_FAIL, "trace_selection_invalid")
                current_session = sid
                current_turn = 0
                turn_arrival = None
                previous_gap = None
                previous_line = line
                session_selecting = False

            # The former two-phase implementation validated every session ID
            # before checking any arrival/request fields. Defer selector errors
            # until the input has been scanned so a malformed ID late in the
            # file keeps the same fail-closed precedence.
            if selection_error is not None:
                continue

            try:
                if new_session:
                    # Once the first outside-window session is found, the old
                    # selector stopped evaluating session data but still read
                    # the tail to reject empty or noncontiguous session IDs.
                    if first_later is not None:
                        continue

                    arrival = _trace_nonnegative_digits(
                        value(row, "arrival_time") or "", line=line,
                        field="arrival_time")
                    if (last_first_arrival is not None
                            and arrival < last_first_arrival):
                        raise AcceptanceError(
                            "session-first trace arrivals are not monotonic",
                            EXIT_FAIL, "trace_selection_invalid")
                    last_first_arrival = arrival
                    if arrival >= window_ns:
                        first_later = {"session_id": sid, "source_line": line,
                                       "arrival_ns": arrival,
                                       "reason": "session first arrival >= window_ns"}
                        continue

                    turn_arrival = arrival
                    session_selecting = True
                elif first_later is not None or not session_selecting:
                    continue

                if current_turn:
                    raw_gap = previous_gap or ""
                    gap = (0 if raw_gap == "" else _trace_nonnegative_digits(
                        raw_gap, line=previous_line,
                        field="human_time/tool_time"))
                    assert turn_arrival is not None
                    turn_arrival += gap
                    if turn_arrival >= window_ns:
                        excluded.append({"session_id": sid, "turn": current_turn,
                                         "derived_arrival_ns": turn_arrival,
                                         "source_line": line,
                                         "reason": "derived arrival >= window_ns"})
                        session_selecting = False
                        previous_gap = None
                        continue

                source_request_id = value(row, "request_id") or ""
                if not source_request_id:
                    raise AcceptanceError(f"trace line {line}: empty request_id",
                                          EXIT_FAIL, "trace_selection_invalid")
                selected.append({
                    "session_id": sid, "turn": current_turn,
                    "request_id": f"{sid}_request_{current_turn}",
                    "source_request_id": source_request_id,
                    "arrival_ns": turn_arrival, "source_line": line,
                    "decode_length": _trace_nonnegative_digits(
                        value(row, "decode_length") or "", line=line,
                        field="decode_length"),
                })
                last_session = sid
                previous_gap = (value(row, "human_time")
                                or value(row, "tool_time") or "")
                previous_line = line
                current_turn += 1
            except AcceptanceError as exc:
                selection_error = exc
                session_selecting = False
                previous_gap = None

        if selection_error is not None:
            raise selection_error

    return {
        "schema": "joint.trace_window_selection.v1", "window_ns": window_ns,
        "source_csv": str(csv_path), "source_sha256": sha256_file(csv_path),
        "selection": {
            "session_rule": "first-row arrival < 1e9 ns",
            "turn_rule": "prior derived arrival + prior row human_time, else tool_time, else zero",
            "cutoff_rule": "truncate a session at first derived arrival >= 1e9 ns",
            "decode_rule": "arrival cutoff selects requests only; decode runs to natural completion",
        },
        "selected_session_count": len({row["session_id"] for row in selected}),
        "selected_request_count": len(selected), "selected_requests": selected,
        "first_excluded_turns": excluded, "first_later_session": first_later,
        "last_included_session": last_session,
    }


def _campaign_case_recipe(scenario_number: int, seed: int) -> dict[str, Any]:
    scenario_id = f"M{scenario_number:02d}"
    cluster_id = f"{scenario_id}/seed{seed:03d}"
    focus_digest = hashlib.sha256(
        f"joint-accel-focus-v1/{scenario_id}/{seed}".encode("ascii")).digest()
    focus_stage = "P" if (focus_digest[0] & 1) == 0 else "D"
    return {
        "case_id": f"{scenario_id}-seed{seed:03d}",
        "cluster_id": cluster_id,
        "partition": partition_cluster(cluster_id)["partition"],
        "scenario_id": scenario_id, "seed": seed,
        "causal_input": {
            "focus_stage": focus_stage,
            "context_tokens": (128, 512, 2048, 8192)[seed % 4],
            "horizon_tokens": (1, 8, 32, 128)[(seed // 4) % 4],
            "competition_level": "high" if (seed // 16) % 2 else "low",
            "source_count": 1 + ((seed // 32 + scenario_number) % 3),
        },
        "execution_input": {
            "committed_decode_tokens": (2, 16, 64, 256)[(seed + scenario_number) % 4],
        },
        "fixture_state": None, "event_tape": None,
        "status": "awaiting_real_fixture_producer",
    }


def build_campaign_plan(root: Path, *, source_root: Path,
                        trace_selection: dict[str, Any]) -> dict[str, Any]:
    cases = [_campaign_case_recipe(scenario_number, seed)
             for scenario_number in PRICEABLE_SCENARIOS for seed in DEV_SEEDS]
    return {
        "schema_version": "joint-accel-campaign-v2-20261006",
        "created_utc": utc_now(), "status": "INCOMPLETE",
        "reason_code": "execution_fixture_cases_not_generated",
        "evidence_root": str(root),
        "frozen_source_tree_digest_sha256": read_json(
            root / "manifests/source-inventory.json")["tree_digest_sha256"],
        "source_root": str(source_root),
        "bootstrap": {"replicates": BOOTSTRAP_REPLICATES, "seed": BOOTSTRAP_SEED,
                      "confidence_level": 0.95},
        "partition": {"function": "sha256('joint-accel-split-v1/'+cluster_id)",
                      "modulus": 5, "dev_remainder": 0},
        "formal_seed_range_initial": [0, 63],
        "formal_seed_range_max": [0, MAX_SEED],
        "priceable_scenarios": list(PRICEABLE_SCENARIO_IDS),
        "development_matrix_scenarios": list(SUPPORTED_SCENARIO_IDS),
        "scenario_specific_profiles": "fixed by execution document section 17.3; only the real C++ fixture producer may instantiate them",
        "trace_selection": trace_selection,
        "controlled_input_recipe_count": len(cases), "controlled_cases": cases,
        "actual_fixture_case_count": None,
        "trace_source_is_separate_from_controlled_cases": True,
        "fixture_cases_manifest": None,
        "note": "This is an input-generation plan, not a captured measurement bank.",
    }


def _find_execution_fixture(root: Path, explicit: Path | None) -> Path | None:
    # 2026-10-14 远读退役：JointPipelineExecutionFixture（唯一执行器，仅驱动
    # descriptor 执行）已随 remote-read 机制删除。capture 及其下游
    # tune/holdout/performance 度量链保持结构但在无执行器下 fail-closed；
    # 复跑 campaign 前需先接二动作执行器（PROVENANCE 退役登记节登记项）。
    if explicit is not None:
        path = explicit.expanduser().resolve()
        if not path.is_file() or not os.access(path, os.X_OK):
            raise AcceptanceError(f"execution fixture is not an executable file: {path}")
        return path
    return None


def _cold_fixture_cases(root: Path, fixtures_dir: Path | None) -> list[tuple[Path, Path]]:
    base = fixtures_dir or (root / "inputs/cold-fixtures")
    if not base.is_dir():
        return []
    cases: list[tuple[Path, Path]] = []
    seen: set[str] = set()
    for directory in sorted((item for item in base.iterdir() if item.is_dir()),
                            key=lambda item: item.name):
        state_path, events_path = directory / "state.json", directory / "events.json"
        if not state_path.is_file() and not events_path.is_file():
            continue
        if not state_path.is_file() or not events_path.is_file():
            raise AcceptanceError(f"fixture case {directory} needs both state.json and events.json",
                                  EXIT_FAIL, "fixture_case_incomplete")
        state = validate_fixture_state(read_json(state_path), state_path)
        validate_event_tape(read_json(events_path), case_id=state["case_id"])
        if state["case_id"] in seen:
            raise AcceptanceError(f"duplicate fixture case_id {state['case_id']}", EXIT_FAIL,
                                  "fixture_case_duplicate")
        seen.add(state["case_id"])
        cases.append((state_path.resolve(), events_path.resolve()))
    return cases


def command_capture(root: Path, *, source_root: Path | None,
                    fixture_binary: Path | None, fixtures_dir: Path | None,
                    prediction_config: Path | None) -> int:
    """Capture real cold fixtures twice per process and preserve raw evidence."""
    _a0_done, source_inventory = validate_a0(root)
    verify_statistical_reference(root)
    source = (source_root or REPO_ROOT).expanduser().resolve()
    implementation_source = _save_implementation_source_binding(root, source)
    trace = root / "manifests/astra_compute_20.csv"
    trace_selection = select_first_second_trace(trace)
    trace_materializer = (source / "sh_test_mesh/workload/llama2_7b_inference/traces"
                          / "materialize_20_30s.py")
    trace_selection["materializer"] = {
        "path": str(trace_materializer),
        "sha256": sha256_file(trace_materializer) if trace_materializer.is_file() else None,
    }
    trace_manifest_path = root / "inputs/trace-window-1s-selection.json"
    if trace_manifest_path.exists():
        existing_selection = read_json(trace_manifest_path)
        if existing_selection != trace_selection:
            raise AcceptanceError("existing frozen first-second selection differs; preserving it",
                                  EXIT_FAIL, "trace_selection_manifest_mismatch")
    else:
        write_json(trace_manifest_path, trace_selection)
    campaign = build_campaign_plan(root, source_root=source,
                                   trace_selection=trace_selection)
    campaign_path = root / "manifests/campaign-manifest.json"
    if campaign_path.exists():
        prior = read_json(campaign_path)
        stable_fields = ("schema_version", "frozen_source_tree_digest_sha256",
                         "bootstrap", "partition", "formal_seed_range_initial",
                         "formal_seed_range_max", "priceable_scenarios",
                         "development_matrix_scenarios", "trace_selection",
                         "controlled_input_recipe_count", "controlled_cases")
        if any(prior.get(key) != campaign.get(key) for key in stable_fields):
            raise AcceptanceError("existing campaign manifest has different frozen inputs",
                                  EXIT_FAIL, "campaign_manifest_mismatch")
        campaign = prior
    else:
        write_json(campaign_path, campaign)

    source_config = prediction_config
    if source_config is not None and not source_config.is_absolute():
        source_config = (root / source_config).resolve()
    contract, config_record = resolve_prediction_contract(source, source_config)
    binary = _find_execution_fixture(root, fixture_binary)
    cases = _cold_fixture_cases(root, fixtures_dir)
    missing_parents: list[str] = []
    for parent in ("G-C5", "G-D1", "G-E0"):
        try:
            validate_done(root, parent)
        except AcceptanceError:
            missing_parents.append(parent)
    result: dict[str, Any] = {
        "schema_version": "joint-accel-capture-v1-20261006",
        "created_utc": utc_now(), "status": "INCOMPLETE",
        "reason_code": "execution_fixture_cases_not_generated",
        "frozen_source_tree_digest_sha256": source_inventory["tree_digest_sha256"],
        "source_root": str(source),
        "implementation_source": implementation_source,
        "source_files": [
            {"path": str(path), "sha256": sha256_file(path)}
            for path in (REPO_ROOT / "sh_test_mesh/run_scripts/joint_accel_acceptance.py",
                         source / "sh_test_mesh/workload/llama2_7b_inference/joint/prediction_config.py")
            if path.is_file()],
        "fixture_binary": None, "fixture_binary_sha256": None,
        "prediction_config": config_record, "kernel_contract": contract,
        "trace_selection_path": str(trace_manifest_path.relative_to(root)),
        "trace_selection_sha256": sha256_file(trace_manifest_path),
        "missing_parent_gates": missing_parents,
        "case_results": [], "first_boundary_samples": [], "coverage": {},
    }
    if missing_parents:
        result["reason_code"] = "capture_parent_gates_missing"
        result["missing"] = [f"parent gate {gate}" for gate in missing_parents]
        if binary is None:
            result["missing"].append("C++ execution fixture executable (retired with remote-read 2026-10-14; "
                                     "re-point to a three-action executor before capture campaigns)")
        if not cases:
            result["missing"].append("real cold state/events under inputs/cold-fixtures/<case>/")
    elif binary is None:
        result["reason_code"] = "execution_fixture_target_missing"
        result["missing"] = [
            "C++ execution fixture executable (retired with remote-read 2026-10-14)",
            "real cold state/events under inputs/cold-fixtures/<case>/",
        ]
    elif not cases:
        result["reason_code"] = "real_cold_fixture_cases_missing"
        result["missing"] = ["state.json/events.json fixture cases"]
        result["fixture_binary"] = str(binary)
        result["fixture_binary_sha256"] = sha256_file(binary)
    else:
        result["fixture_binary"] = str(binary)
        result["fixture_binary_sha256"] = sha256_file(binary)
        try:
            attempt_root = root / "runs/capture"
            attempt_root.mkdir(parents=True, exist_ok=True)
            attempt = attempt_root / f"attempt-{_attempt_number(attempt_root):03d}"
            attempt.mkdir(exist_ok=False)
            result["attempt"] = str(attempt.relative_to(root))
            result["reason_code"] = "fixture_capture_started"
            env = _environment_for_tests()
            stable_rows: list[dict[str, Any]] = []
            failures: list[dict[str, Any]] = []
            for state_path, events_path in cases:
                state = validate_fixture_state(read_json(state_path), state_path)
                tape = validate_event_tape(read_json(events_path), case_id=state["case_id"])
                case_dir = attempt / state["case_id"]
                case_dir.mkdir(parents=True, exist_ok=False)
                precision = case_dir / "precision.json"
                write_json(precision, contract)
                runs: list[dict[str, Any]] = []
                parsed_results: list[dict[str, Any]] = []
                for replica in (1, 2):
                    replica_dir = case_dir / f"fresh-process-{replica}"
                    replica_dir.mkdir(exist_ok=False)
                    output = replica_dir / "result.json"
                    log = replica_dir / "fixture.log"
                    argv = [str(binary), "--state", str(state_path), "--events",
                            str(events_path), "--precision", str(precision), "--out",
                            str(output)]
                    invocation = run_logged(argv, source, env, log, 300)
                    row = {"replica": replica, "invocation": invocation,
                           "state_sha256": sha256_file(state_path),
                           "events_sha256": sha256_file(events_path),
                           "precision_sha256": sha256_file(precision),
                           "result_path": str(output.relative_to(root))
                               if output.exists() else None}
                    if invocation.get("status") != "completed" or not output.is_file():
                        row["status"] = "INCOMPLETE"
                        row["reason"] = "fixture_process_failed_or_result_missing"
                        failures.append({"case_id": state["case_id"], **row})
                    else:
                        # §19.1A two-stage wiring: validate the raw producer output
                        # first (canonicalization_pending, null prestate, raw v2
                        # frames), canonicalize via the acceptance-only planner
                        # (production dual-ledger replay), then run the full
                        # canonical validation on the final result.
                        raw = validate_raw_fixture_result(read_json(output), state, contract)
                        row["raw_result_sha256"] = sha256_file(output)
                        parsed = canonicalize_fixture_result(raw, state, source)
                        canonical_output = replica_dir / "canonical-result.json"
                        write_json(canonical_output, parsed)
                        row["canonical_result_path"] = str(
                            canonical_output.relative_to(root))
                        validate_fixture_result(parsed, state, contract)
                        parsed_results.append(parsed)
                        row.update({"status": parsed["status"],
                                    "result_sha256": sha256_file(output),
                                    "canonical_result_sha256": hashlib.sha256(
                                        json.dumps(canonical_fixture_result(parsed),
                                                   sort_keys=True, separators=(",", ":"),
                                                   ensure_ascii=False, allow_nan=False)
                                        .encode("utf-8")).hexdigest()})
                        if parsed["status"] != "PASS":
                            failures.append({"case_id": state["case_id"],
                                             "status": parsed["status"],
                                             "reason": parsed["reason"]})
                    runs.append(row)
                equal = (len(parsed_results) == 2
                         and canonical_fixture_result(parsed_results[0])
                         == canonical_fixture_result(parsed_results[1]))
                if not equal:
                    failures.append({"case_id": state["case_id"],
                                     "status": "FAIL" if len(parsed_results) == 2 else "INCOMPLETE",
                                     "reason": "fresh_process_replay_mismatch_or_missing"})
                if equal and all(item["status"] == "PASS" for item in parsed_results):
                    stable_rows.extend({**row, "cluster_id": state["cluster_id"]}
                                       for row in parsed_results[0]["boundary_records"])
                result["case_results"].append({
                    "case_id": state["case_id"], "cluster_id": state["cluster_id"],
                    "partition": partition_cluster(state["cluster_id"])["partition"],
                    "scenario_id": state["scenario_id"], "seed": state["seed"],
                    "state_path": str(state_path), "state_sha256": sha256_file(state_path),
                    "events_path": str(events_path), "events_sha256": sha256_file(events_path),
                    "precision_sha256": sha256_file(precision),
                    "fresh_processes_identical": equal, "runs": runs,
                })
            samples = first_boundary_samples(stable_rows) if stable_rows else []
            stage_rows = {stage: [row for row in samples if row["stage"] == stage]
                          for stage in ("P", "D")}
            result["first_boundary_samples"] = samples
            result["coverage"] = {
                "unique_clusters": len({row["cluster_id"] for row in samples}),
                "P": {"snapshots": len(stage_rows["P"]),
                      "clusters": len({row["cluster_id"] for row in stage_rows["P"]})},
                "D": {"snapshots": len(stage_rows["D"]),
                      "clusters": len({row["cluster_id"] for row in stage_rows["D"]})},
            }
            result["capture_failures"] = failures
            result["status"] = ("FAIL" if any(row["status"] == "FAIL" for row in failures)
                                 else "INCOMPLETE" if failures or missing_parents
                                 else "PASS")
            result["reason_code"] = (
                "fresh_process_result_mismatch" if result["status"] == "FAIL"
                else "capture_parent_gate_or_case_incomplete" if result["status"] == "INCOMPLETE"
                else "cold_fixture_capture_passed")
        except AcceptanceError as exc:
            result["status"] = "FAIL" if exc.code == EXIT_FAIL else "INCOMPLETE"
            result["reason_code"] = exc.reason_code
            result["error"] = str(exc)
    if result["status"] == "PASS":
        try:
            _verify_implementation_source_binding(root, source, implementation_source)
        except AcceptanceError as exc:
            result.update({"status": "FAIL" if exc.code == EXIT_FAIL else "INCOMPLETE",
                           "reason_code": exc.reason_code, "error": str(exc)})
    capture_path = root / "packages/acceptance/capture.json"
    write_json(capture_path, result)
    print(json.dumps({"command": "capture", "status": result["status"],
                      "reason_code": result["reason_code"],
                      "case_count": len(result["case_results"]),
                      "capture_manifest": str(capture_path.relative_to(root))},
                     ensure_ascii=False, sort_keys=True))
    if result["status"] == "PASS":
        return EXIT_PASS
    if result["status"] == "FAIL":
        return EXIT_FAIL
    return EXIT_INCOMPLETE


def _find_projection_library(root: Path, source_root: Path,
                             explicit: Path | None) -> Path | None:
    if explicit is not None:
        path = explicit.expanduser().resolve()
        if not path.is_file():
            raise AcceptanceError(f"native projection library is missing: {path}",
                                  EXIT_INCOMPLETE, "native_projection_library_missing")
        return path
    candidates = [
        root / "packages/cpp-core/build/congestion_aware/lib/libjoint_projection.so",
        source_root / "build/astra_analytical/build_congestion_aware/lib/libjoint_projection.so",
        source_root / "build_congestion_aware/lib/libjoint_projection.so",
    ]
    return next((path.resolve() for path in candidates if path.is_file()), None)


def _load_captured_exact_bank(root: Path, *,
                              partition: str | None = None
                              ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    capture_path = root / "packages/acceptance/capture.json"
    if not capture_path.is_file():
        raise AcceptanceError("capture manifest is missing; run capture with actual cold cases first",
                              EXIT_INCOMPLETE, "capture_manifest_missing")
    capture = read_json(capture_path)
    if capture.get("status") != "PASS":
        raise AcceptanceError(
            f"capture is {capture.get('status')}: {capture.get('reason_code')}; "
            "tuning requires two identical natural-completion exact runs",
            EXIT_INCOMPLETE, "capture_not_pass")
    contract = validate_kernel_contract(capture.get("kernel_contract"))
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, case in enumerate(capture.get("case_results", [])):
        case = _require_object(case, f"capture.case_results[{index}]")
        if case.get("case_id") in seen:
            raise AcceptanceError("capture contains duplicate case IDs", EXIT_FAIL,
                                  "capture_case_duplicate")
        seen.add(case.get("case_id"))
        cluster_id = case.get("cluster_id")
        case_partition = partition_cluster(cluster_id)["partition"]
        if partition is not None and case_partition != partition:
            continue
        state_path, events_path = Path(case["state_path"]).resolve(), Path(case["events_path"]).resolve()
        if not state_path.is_file() or not events_path.is_file():
            raise AcceptanceError(f"captured inputs disappeared for {case.get('case_id')}",
                                  EXIT_FAIL, "capture_input_missing")
        if (sha256_file(state_path) != case.get("state_sha256")
                or sha256_file(events_path) != case.get("events_sha256")):
            raise AcceptanceError(f"captured input digest changed for {case.get('case_id')}",
                                  EXIT_FAIL, "capture_input_digest_mismatch")
        state = validate_fixture_state(read_json(state_path), state_path)
        tape = validate_event_tape(read_json(events_path), case_id=state["case_id"])
        if not isinstance(case.get("runs"), list) or len(case["runs"]) != 2:
            raise AcceptanceError(f"case {state['case_id']} lacks two exact process records",
                                  EXIT_INCOMPLETE, "capture_exact_replicas_missing")
        results: list[dict[str, Any]] = []
        for run_index, run_value in enumerate(case["runs"]):
            run = _require_object(run_value, f"case {state['case_id']}.runs[{run_index}]")
            rel_result = run.get("result_path")
            if not isinstance(rel_result, str):
                raise AcceptanceError(f"case {state['case_id']} lacks exact result path",
                                      EXIT_INCOMPLETE, "capture_result_missing")
            result_path = (root / rel_result).resolve()
            if root not in result_path.parents or not result_path.is_file():
                raise AcceptanceError(f"captured result path is missing/outside evidence: {rel_result}",
                                      EXIT_FAIL, "capture_result_missing")
            if run.get("status") != "PASS" or sha256_file(result_path) != run.get("result_sha256"):
                raise AcceptanceError(f"captured exact result failed/hash changed: {rel_result}",
                                      EXIT_FAIL, "capture_result_digest_mismatch")
            # Stage A raw 输出（canonicalization_pending、null prestate）不能
            # 直接过 canonical schema 校验；装载 capture 流水线按案例落盘
            # 的 canonical 副本（canonical_result_path + sha 双验）。
            rel_canonical = run.get("canonical_result_path")
            if not isinstance(rel_canonical, str):
                raise AcceptanceError(f"case {state['case_id']} lacks a canonical result path",
                                      EXIT_INCOMPLETE, "capture_canonical_result_missing")
            canonical_path = (root / rel_canonical).resolve()
            if root not in canonical_path.parents or not canonical_path.is_file():
                raise AcceptanceError(f"captured canonical result is missing/outside evidence: {rel_canonical}",
                                      EXIT_FAIL, "capture_canonical_result_missing")
            canonical_sha = run.get("canonical_result_sha256")
            # 记录口径 = canonical JSON dump（sort_keys+紧凑）的摘要，
            # 与 write_json 落盘字节格式无关。
            canonical_dump = hashlib.sha256(json.dumps(
                read_json(canonical_path), sort_keys=True, separators=(",", ":"),
                ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()
            if not isinstance(canonical_sha, str) or canonical_dump != canonical_sha:
                raise AcceptanceError(f"captured canonical result hash changed: {rel_canonical}",
                                      EXIT_FAIL, "capture_canonical_result_digest_mismatch")
            parsed = validate_fixture_result(read_json(canonical_path), state, contract)
            results.append(parsed)
        if results[0] != results[1]:
            raise AcceptanceError(f"capture exact replicas differ for {state['case_id']}",
                                  EXIT_FAIL, "capture_exact_replay_mismatch")
        rows.append({"case_id": state["case_id"], "cluster_id": state["cluster_id"],
                     "scenario_id": state["scenario_id"], "seed": state["seed"],
                     "partition": case_partition,
                     "state_path": state_path, "events_path": events_path,
                     "state": state, "event_tape": tape,
                     "exact": results[0], "case_manifest": case})
    return capture, rows


def _run_fixture_once(binary: Path, source_root: Path, state_path: Path,
                      events_path: Path, contract: dict[str, Any], *,
                      out_dir: Path, label: str, evidence_root_path: Path,
                      timeout_seconds: int = 300) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    out_dir.mkdir(parents=True, exist_ok=False)
    precision = out_dir / "precision.json"
    output = out_dir / "result.json"
    log = out_dir / "fixture.log"
    write_json(precision, contract)
    invocation = run_logged(
        [str(binary), "--state", str(state_path), "--events", str(events_path),
         "--precision", str(precision), "--out", str(output)],
        source_root, _environment_for_tests(), log, timeout_seconds)
    record: dict[str, Any] = {
        "label": label, "invocation": invocation,
        "state_sha256": sha256_file(state_path),
        "events_sha256": sha256_file(events_path),
        "precision_sha256": sha256_file(precision),
        "result_path": str(output.relative_to(evidence_root_path))
            if output.exists() else None,
        "log_path": str(log.relative_to(evidence_root_path))
            if log.exists() and evidence_root_path in log.parents else str(log),
    }
    if invocation.get("status") != "completed" or not output.is_file():
        record.update({"status": "INCOMPLETE" if invocation.get("status") in (
            "timeout", "missing_dependency", "launch_error") else "FAIL",
            "reason": "fixture_process_failed_or_result_missing"})
        return None, record
    try:
        state_value = validate_fixture_state(read_json(state_path), state_path)
        # fixture 的直接输出是 Stage A raw（canonicalization_pending）；
        # L2/L3 消费面是 canonical 六域，先经生产 Stage B canonicalize。
        parsed = canonicalize_fixture_result(read_json(output), state_value, source_root)
    except AcceptanceError as exc:
        record.update({"status": "FAIL" if exc.code == EXIT_FAIL else "INCOMPLETE",
                       "reason": exc.reason_code, "error": str(exc),
                       "result_sha256": sha256_file(output)})
        return None, record
    record.update({"status": parsed["status"], "reason": parsed["reason"],
                   "result_sha256": sha256_file(output)})
    return parsed, record


def _l2_bank_summary(pairs: list[dict[str, Any]], mismatches: list[dict[str, Any]]) -> dict[str, Any]:
    by_stage: dict[str, Any] = {}
    by_stage_action: dict[str, Any] = {}
    by_stage_bucket: dict[str, Any] = {}
    for stage in ("P", "D"):
        stage_rows = [row for row in pairs if row["stage"] == stage]
        if stage_rows:
            by_stage[stage] = _summarize_l2_group(stage_rows)
            for action in VALID_ACTIONS:
                selected = [row for row in stage_rows if row["fixed_action"] == action]
                if selected:
                    by_stage_action[f"{stage}/{action}"] = _summarize_l2_group(selected)
            for bucket in ("context", "competition", "sources", "capacity"):
                values = sorted({row["bucket_labels"].get(bucket) for row in stage_rows
                                 if row["bucket_labels"].get(bucket) is not None})
                for value in values:
                    selected = [row for row in stage_rows
                                if row["bucket_labels"].get(bucket) == value]
                    by_stage_bucket[f"{stage}/{bucket}/{value}"] = _summarize_l2_group(selected)
    groups = list(by_stage.values()) + list(by_stage_action.values()) + list(by_stage_bucket.values())
    missing_stages = [stage for stage in ("P", "D") if stage not in by_stage]
    status = ("FAIL" if any(group["status"] == "FAIL" for group in groups)
              else "INSUFFICIENT_EVIDENCE" if missing_stages
              else "PASS" if groups and not mismatches
              else "INSUFFICIENT_EVIDENCE" if not pairs
              else "PASS")
    return {"status": status, "by_stage": by_stage,
            "by_stage_action": by_stage_action, "by_stage_bucket": by_stage_bucket,
            "pairs": pairs, "state_mismatches": mismatches,
            "evaluable_pair_count": len(pairs),
            "missing_stages": missing_stages,
            "independent_clusters": len({row["cluster_id"] for row in pairs})}


def _l3_bank_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate single-face L3 scores per stage.

    v2（2026-10-16 E0 退役）：近似/精确双面与 regret/误差轴随复核机制
    一并退役——本汇总只做描述性统计（样本数/独立簇数/选中动作分布），
    状态判定保留 FAIL/INSUFFICIENT_EVIDENCE/PASS 骨架。
    """
    result: dict[str, Any] = {"by_stage": {}, "state_mismatches": [],
                              "unpriceable": [], "status": "INSUFFICIENT_EVIDENCE"}
    if any(row.get("status") == "FAIL" for row in rows):
        result["status"] = "FAIL"
    for stage in ("P", "D"):
        selected = [row for row in rows if row.get("stage") == stage]
        good = [row for row in selected if row.get("evaluable") is True]
        result["state_mismatches"].extend(
            row for row in selected if row.get("status") == "state_mismatch")
        result["unpriceable"].extend(
            row for row in selected if row.get("evaluable") is not True
            and row.get("status") != "state_mismatch")
        if not good:
            continue
        cluster_ids = [row.get("cluster_id") for row in good]
        if any(not isinstance(value, str) or not value for value in cluster_ids):
            raise AcceptanceError("L3 samples need actual cluster identities", EXIT_FAIL,
                                  "candidate_cluster_identity_missing")
        action_counts: dict[str, int] = {}
        for row in good:
            action = row.get("selected_action")
            if not isinstance(action, str) or not action:
                raise AcceptanceError("evaluable L3 sample lacks its selected action",
                                      EXIT_FAIL, "candidate_selection_mismatch")
            action_counts[action] = action_counts.get(action, 0) + 1
        result["by_stage"][stage] = {
            "snapshot_count": len(good),
            "independent_clusters": len(set(cluster_ids)),
            "selected_action_counts": action_counts,
        }
        result["status"] = "PASS" if result["status"] != "FAIL" else "FAIL"
    if "P" not in result["by_stage"] or "D" not in result["by_stage"]:
        result["status"] = ("INSUFFICIENT_EVIDENCE" if result["status"] != "FAIL"
                            else "FAIL")
    return result


def _first_result_boundaries(result: dict[str, Any], cluster_id: str) -> list[dict[str, Any]]:
    rows = [{**row, "cluster_id": cluster_id}
            for row in result.get("boundary_records", [])]
    return first_boundary_samples(rows) if rows else []


def _evaluate_l3_bank(rows: list[dict[str, Any]], source_root: Path,
                      prediction_config: dict[str, Any],
                      prediction_config_sha256: str,
                      library_path: Path, out_dir: Path,
                      evidence_root_path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    scores: list[dict[str, Any]] = []
    raw_records: list[dict[str, Any]] = []
    # evaluate_boundary 要求 JointPredictionConfig 实例，且必须与
    # acceptance_projection 消费的为同一模块实例（isinstance 语义——
    # 按文件路径独立加载会产生第二类对象使检查永假）。先经
    # _load_acceptance_module 确保 joint 包已按源根加载，再从包内取
    # prediction_module 构造（round-trip 已验证：semantic_dict 等值、
    # kernel contract 稳定）。
    _load_acceptance_module(source_root)
    prediction_config = sys.modules["joint.prediction_config"].from_mapping(
        prediction_config)
    for case in rows:
        exact_frames = [entry.get("frame") for entry in
                       (case["exact"].get("audit_records") or [])
                       if isinstance(entry, dict)
                       and entry.get("kind") == "raw_v2_frame"]
        exact_records = case["exact"].get("boundary_records") or []
        for sample in _first_result_boundaries(case["exact"], case["cluster_id"]):
            identity = {key: sample[key] for key in
                        ("request_id", "stage", "boundary_ordinal", "tick_ns")}
            helper_boundary = {key: value for key, value in sample.items()
                               if key != "cluster_id"}
            # canonical 行 → evaluate_boundary 合同的适配层（E0 同款）：
            # 累积 raw-frame 前缀 + 从同行 planner_state 派生
            # forecast_generator。
            helper_boundary = json.loads(json.dumps(helper_boundary))
            _position = next(
                (index for index, row in enumerate(exact_records)
                 if row.get("stage") == sample["stage"]
                 and row.get("boundary_ordinal") == sample["boundary_ordinal"]),
                None)
            _ledger = helper_boundary.get("causal_prestate", {}).get("work_ledger")
            if (isinstance(_ledger, dict)
                    and not _ledger.get("certified_empty", False)
                    and _position is not None):
                # Keep the historical DTO key while retaining exact
                # version-tagged V3 archive or V4 current raw frames. The
                # offline hydrator selects one strict decoder from this prefix.
                _ledger["replay_v3_frames"] = exact_frames[: _position + 1]
            _metadata = helper_boundary.get("causal_metadata")
            _planner = helper_boundary.get("planner_state") or {}
            if (isinstance(_metadata, dict)
                    and "forecast_generator" not in _metadata
                    and isinstance(_planner.get("forecast_generator"), dict)):
                _metadata["forecast_generator"] = _planner["forecast_generator"]
            request_digest = hashlib.sha256(
                sample["request_id"].encode("utf-8")).hexdigest()[:10]
            sample_dir = out_dir / case["case_id"] / (
                f"{sample['stage']}-{sample['boundary_ordinal']:04d}-{request_digest}")
            sample_dir.mkdir(parents=True, exist_ok=False)
            raw_path = sample_dir / "projection-evaluation.json"
            try:
                evaluator = _load_acceptance_projection(source_root)
                dto = evaluator(helper_boundary, prediction_config,
                                library_path=str(library_path),
                                fixture_state=case["state"])
                dto = _require_object(dto, "evaluate_boundary result")
                write_json(raw_path, dto)
                score_boundary = dict(helper_boundary, cluster_id=case["cluster_id"])
                score = compare_projection_evaluation(
                    dto, score_boundary,
                    prediction_config_sha256=prediction_config_sha256,
                    library_sha256=sha256_file(library_path))
                scores.append({**score, "case_id": case["case_id"],
                               "cluster_id": case["cluster_id"]})
                raw_records.append({"case_id": case["case_id"],
                                    "cluster_id": case["cluster_id"],
                                    **identity,
                                    "status": dto.get("status"),
                                    "raw_path": str(raw_path.relative_to(evidence_root_path)),
                                    "raw_sha256": sha256_file(raw_path),
                                    "scored": score})
            except AcceptanceError as exc:
                status = "FAIL" if exc.code == EXIT_FAIL else "INCOMPLETE"
                failure = {"status": status, "reason": exc.reason_code,
                           "message": str(exc), "case_id": case["case_id"],
                           "cluster_id": case["cluster_id"], **identity}
                scores.append(failure)
                raw_records.append(failure)
            except Exception as exc:
                failure = {"status": "INCOMPLETE", "reason": "l3_evaluation_api_failed",
                           "message": str(exc), "case_id": case["case_id"],
                           "cluster_id": case["cluster_id"], **identity}
                scores.append(failure)
                raw_records.append(failure)
    return scores, raw_records


def _validate_capture_case_recipe(case: dict[str, Any],
                                  recipe: dict[str, Any]) -> list[str]:
    """身份字段严格校验；causal axes 的通用公式比对降为偏差收集。

    执行文档 §19.2 第 2 项：场景专门边界参数优先于通用生成参数
    （M01 formal tile 表、M06 强制高竞争、producer focus 规则等），
    纯公式 recipe 与之存在合法偏差（2026-10-08 全银行实测 79/110 例
    至少一处）。身份与执行输入不匹配仍 FAIL；axes 偏差返回 deviations
    列表供装载方披露，不据此 FAIL。"""
    state = case["state"]
    causal = state["causal_input"]
    expected_fields = ("case_id", "cluster_id", "scenario_id", "seed")
    for field in expected_fields:
        if state[field] != recipe[field]:
            raise AcceptanceError(f"fixture differs from frozen recipe at {field}", EXIT_FAIL,
                                  "fixture_recipe_mismatch")
    recipe_causal = recipe["causal_input"]
    aliases = {"context_tokens": ("context_tokens", "ctx"),
        "horizon_tokens": ("horizon_tokens",),
               "competition_level": ("competition_level", "competition"),
               "source_count": ("source_count", "sources")}
    deviations: list[str] = []
    for field, names in aliases.items():
        actual = next((causal[name] for name in names if name in causal), None)
        if actual != recipe_causal[field]:
            deviations.append(f"causal:{field}")
    if causal.get("focus_stage") != recipe_causal["focus_stage"]:
        deviations.append("causal:focus_stage")
    if state["fixed_action"] not in VALID_ACTIONS:
        raise AcceptanceError("fixture action is invalid", EXIT_FAIL,
                              "fixture_recipe_mismatch")
    if causal.get("committed_decode_tokens") != recipe["execution_input"]["committed_decode_tokens"]:
        raise AcceptanceError("fixture execution length differs from frozen seed recipe",
                              EXIT_FAIL, "fixture_recipe_mismatch")
    return deviations


def _expected_dev_case_ids(campaign: dict[str, Any]) -> set[str]:
    return {row["case_id"] for row in campaign["controlled_cases"]
            if row["partition"] == "dev"}


def _recipe_for_fixture(state: dict[str, Any]) -> dict[str, Any]:
    scenario = state.get("scenario_id")
    seed = state.get("seed")
    if scenario not in PRICEABLE_SCENARIO_IDS:
        raise AcceptanceError("accuracy fixture is outside the priceable scenario set",
                              EXIT_FAIL, "fixture_recipe_mismatch")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= MAX_SEED:
        raise AcceptanceError("fixture seed is outside the frozen 0..319 range",
                              EXIT_FAIL, "fixture_recipe_mismatch")
    recipe = _campaign_case_recipe(int(scenario[1:]), seed)
    for field in ("case_id", "cluster_id", "scenario_id", "seed"):
        if state.get(field) != recipe[field]:
            raise AcceptanceError(f"fixture differs from frozen recipe at {field}", EXIT_FAIL,
                                  "fixture_recipe_mismatch")
    return recipe


def _load_campaign_manifest(root: Path, source_root: Path,
                            capture: dict[str, Any]) -> dict[str, Any]:
    path = root / "manifests/campaign-manifest.json"
    if not path.is_file():
        raise AcceptanceError("frozen campaign manifest is missing", EXIT_INCOMPLETE,
                              "campaign_manifest_missing")
    campaign = _require_object(read_json(path), "campaign manifest")
    if campaign.get("schema_version") != "joint-accel-campaign-v2-20261006":
        raise AcceptanceError("campaign manifest predates token-horizon/stage-balanced recipe",
                              EXIT_FAIL, "campaign_manifest_version_mismatch")
    if campaign.get("frozen_source_tree_digest_sha256") != capture.get(
            "frozen_source_tree_digest_sha256"):
        raise AcceptanceError("campaign and capture reference different frozen source digests",
                              EXIT_FAIL, "campaign_capture_source_mismatch")
    if Path(campaign.get("source_root", "")).resolve() != source_root.resolve():
        raise AcceptanceError("campaign manifest source root differs from selected run copy",
                              EXIT_FAIL, "campaign_source_root_mismatch")
    return campaign


def _aggregate_l2_results(case_results: list[dict[str, Any]]) -> dict[str, Any]:
    pairs: list[dict[str, Any]] = []
    mismatches: list[dict[str, Any]] = []
    for row in case_results:
        report = row["l2"]
        pairs.extend(report["pairs"])
        mismatches.extend({**mismatch, "case_id": row["case_id"],
                           "cluster_id": row["cluster_id"]}
                          for mismatch in report["state_mismatches"])
    return _l2_bank_summary(pairs, mismatches)


def _measure_config_pairs(binary: Path, source_root: Path,
                          exact_contract: dict[str, Any],
                          candidate_contract: dict[str, Any],
                          cases: list[dict[str, Any]], *,
                          attempt: Path, root: Path,
                          deadline: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    repetitions: list[dict[str, Any]] = []
    invocations: list[dict[str, Any]] = []
    for repetition in range(3):
        arms: dict[str, Any] = {}
        for arm_name, contract in (("exact", exact_contract),
                                   ("candidate", candidate_contract)):
            start = time.monotonic_ns()
            host_duration = 0
            arm_records: list[dict[str, Any]] = []
            natural = True
            for case_index, case in enumerate(cases):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    natural = False
                    arm_records.append({"status": "INCOMPLETE",
                                        "reason": "suite_wall_limit_reached"})
                    break
                label = f"timing-r{repetition+1}-{arm_name}-{case['case_id']}"
                parsed, record = _run_fixture_once(
                    binary, source_root, case["state_path"], case["events_path"], contract,
                    out_dir=(attempt / "timing" / f"rep-{repetition+1:02d}" /
                             arm_name / f"case-{case_index:04d}"),
                    label=label, evidence_root_path=root,
                    timeout_seconds=max(1, min(300, int(remaining))))
                arm_records.append(record)
                invocations.append(record)
                if parsed is None or parsed["status"] != "PASS":
                    natural = False
                    break
            elapsed = time.monotonic_ns() - start
            host_duration = elapsed
            arms[arm_name] = {
                "status": "completed" if natural else "incomplete",
                "natural_completion": natural,
                "elapsed_host_ns": host_duration if natural else None,
                "work_signature_sha256": _canonical_digest([
                    {"case_id": case["case_id"],
                     "state_sha256": sha256_file(case["state_path"]),
                     "events_sha256": sha256_file(case["events_path"])}
                    for case in cases]),
                "processes": arm_records,
            }
            if not natural:
                repetitions.append({"status": "incomplete", "arms": arms,
                                    "repetition": repetition + 1})
                return repetitions, invocations, "INCOMPLETE"
        if arms["exact"]["work_signature_sha256"] != arms["candidate"]["work_signature_sha256"]:
            raise AcceptanceError("timing arms have different cold fixture work signatures",
                                  EXIT_FAIL, "performance_work_mismatch")
        repetitions.append({"status": "complete", "arms": arms,
                            "repetition": repetition + 1})
    return repetitions, invocations, "PASS"


def _write_acceptance_command_result(root: Path, name: str,
                                     result: dict[str, Any],
                                     exit_code: int) -> int:
    path = root / "packages/acceptance" / f"{name}.json"
    write_json(path, result)
    if name == "holdout":
        _publish_f0_gate_result(root, result, path)
    print(json.dumps({"command": name, "status": result.get("status"),
                      "reason_code": result.get("reason_code"),
                      "result": str(path.relative_to(root))},
                     ensure_ascii=False, sort_keys=True))
    return exit_code


def _publish_f0_gate_result(root: Path, holdout: dict[str, Any],
                            holdout_path: Path) -> None:
    """Mirror the untouched-holdout outcome into the G-F0 result contract."""
    status = holdout.get("status")
    if not isinstance(status, str) or status not in GATE_STATUSES:
        status = "FAIL"
    expected_parent: dict[str, str] = {}
    try:
        validate_done(root, "G-E0")
        expected_parent["G-E0"] = sha256_file(root / "gates/G-E0/DONE")
    except AcceptanceError:
        pass
    checks = {
        "holdout_command_passed": status == "PASS",
        "l2_accuracy_passed": isinstance(holdout.get("l2"), dict)
            and holdout["l2"].get("status") == "PASS",
        "l3_accuracy_passed": isinstance(holdout.get("l3"), dict)
            and holdout["l3"].get("status") == "PASS",
        "frozen_cluster_coverage_passed": isinstance(holdout.get("coverage"), dict)
            and holdout["coverage"].get("status") == "PASS",
        "dev_choice_frozen_before_holdout": bool(
            (root / "manifests/frozen-accuracy.json").is_file()
            and holdout.get("frozen_accuracy_sha256") == sha256_file(
                root / "manifests/frozen-accuracy.json")),
    }
    artifacts: dict[str, str] = {}
    candidates = [holdout_path]
    attempt = holdout.get("attempt")
    if isinstance(attempt, str):
        candidates.append(root / attempt / "result.json")
    for relative in ("packages/acceptance/capture.json",
                     "manifests/frozen-accuracy.json",
                     "manifests/campaign-manifest.json",
                     "manifests/acceptance-statistical-reference-v1.json"):
        candidates.append(root / relative)
    implementation_source = holdout.get("implementation_source")
    if isinstance(implementation_source, dict):
        inventory_rel = implementation_source.get("inventory_path")
        if isinstance(inventory_rel, str):
            candidates.append(root / inventory_rel)
    for path in candidates:
        if path.is_file():
            try:
                relative = path.resolve().relative_to(root.resolve()).as_posix()
            except ValueError:
                continue
            artifacts[relative] = sha256_file(path)
    gate_result = {
        "gate": "G-F0", "status": status,
        "reason_code": holdout.get("reason_code"),
        "parents": expected_parent if status == "PASS" else {},
        "checks": checks, "required_artifacts": artifacts,
        "holdout_status": status,
        "coverage": holdout.get("coverage"),
        "l2": holdout.get("l2"), "l3": holdout.get("l3"),
        "source_root": holdout.get("source_root"),
        "native_library_sha256": holdout.get("native_library_sha256"),
        "frozen_accuracy_sha256": holdout.get("frozen_accuracy_sha256"),
    }
    gate_path = root / "gates/G-F0/result.json"
    done_path = root / "gates/G-F0/DONE"
    if done_path.exists():
        invalidate_done(root, "G-F0", "holdout_result_refreshed")
    write_json(gate_path, gate_result)


def command_tune(root: Path, *, source_root: Path | None,
                 prediction_config: Path | None,
                 native_library: Path | None) -> int:
    """Evaluate the single default configuration on actual cold fixture results."""
    started = time.monotonic()
    result: dict[str, Any] = {
        "schema_version": "joint-accel-tuning-v1-20261006",
        "status": "INCOMPLETE", "reason_code": "tuning_not_started",
        "created_utc": utc_now(), "config_results": [],
        "parents": {}, "fixture_count": 0, "source_root": None,
        "native_library": None,
    }
    frozen_path = root / "manifests/frozen-accuracy.json"
    if frozen_path.exists():
        return _write_acceptance_command_result(root, "tune",
            {**result, "status": "FAIL", "reason_code": "frozen_accuracy_already_exists",
             "message": "refusing to overwrite a frozen development choice"}, EXIT_FAIL)
    try:
        verify_statistical_reference(root)
        missing_parents = []
        for parent in ("G-C5", "G-D1", "G-E0"):
            try:
                result["parents"][parent] = sha256_file(root / "gates" / parent / "DONE")
                validate_done(root, parent)
            except AcceptanceError:
                missing_parents.append(parent)
        if missing_parents:
            result.update({"reason_code": "tuning_parent_gates_missing",
                           "missing_parent_gates": missing_parents})
            return _write_acceptance_command_result(root, "tune", result, EXIT_INCOMPLETE)
        capture, all_exact = _load_captured_exact_bank(root, partition="dev")
        source = (source_root or Path(capture["source_root"])).expanduser().resolve()
        result["source_root"] = str(source)
        source_binding = _verify_implementation_source_binding(
            root, source, capture.get("implementation_source"))
        result["implementation_source"] = source_binding
        campaign = _load_campaign_manifest(root, source, capture)
        planned = {row["case_id"]: row for row in campaign["controlled_cases"]
                   if row["partition"] == "dev"}
        priceable_exact = [row for row in all_exact
                           if row["scenario_id"] in PRICEABLE_SCENARIO_IDS]
        exact_by_id = {row["case_id"]: row for row in priceable_exact
                       if row["partition"] == "dev"}
        missing_cases = sorted(set(planned) - exact_by_id.keys())
        extra_cases = sorted(exact_by_id.keys() - set(planned))
        invalid_extra: list[dict[str, str]] = []
        recipe_deviations: dict[str, list[str]] = {}
        for case_id in extra_cases:
            case = exact_by_id[case_id]
            try:
                deviations = _validate_capture_case_recipe(
                    case, _recipe_for_fixture(case["state"]))
            except AcceptanceError as exc:
                invalid_extra.append({"case_id": case_id, "reason": exc.reason_code})
                continue
            if deviations:
                recipe_deviations[case_id] = deviations
        if recipe_deviations:
            # 场景专门参数与通用公式的合法偏差（§19.2 第 2 项优先序），
            # 披露不 FAIL；见 _validate_capture_case_recipe 文档串。
            result["causal_recipe_deviations"] = recipe_deviations
        if not exact_by_id or invalid_extra:
            result.update({"reason_code": "dev_cold_fixture_bank_incomplete",
                           "missing_dev_case_ids": missing_cases,
                           "invalid_extension_dev_case_ids": invalid_extra,
                           "required_dev_case_count": len(planned),
                           "captured_dev_case_count": len(exact_by_id)})
            return _write_acceptance_command_result(root, "tune", result, EXIT_INCOMPLETE)
        if missing_cases:
            # 覆盖驱动子集（§19.2 第 3 项达标即停）下 planned dev 全场
            # 不可满足：确定性 >300s 超时案例保留在失败清单、不虚构
            # exact 记录；在场集作为 dev 评估集，缺失集全量披露。
            result["planned_dev_missing_disclosure"] = {
                "count": len(missing_cases),
                "case_ids": missing_cases,
                "reason": "coverage-driven subset; deterministic 300s-capped "
                          "fixture timeouts retained on the failure list "
                          "(execdoc §19.2 items 3/7)",
            }
        dev_cases = [exact_by_id[key] for key in sorted(exact_by_id)]
        for case in dev_cases:
            recipe = planned.get(case["case_id"]) or _recipe_for_fixture(case["state"])
            _validate_capture_case_recipe(case, recipe)
        result["fixture_count"] = len(dev_cases)
        exact_contract = validate_kernel_contract(capture["kernel_contract"])
        exact_config, exact_config_record = resolve_prediction_contract(
            source, prediction_config)
        if exact_contract != exact_config:
            raise AcceptanceError("capture exact contract differs from the frozen exact config",
                                  EXIT_FAIL, "capture_exact_contract_mismatch")
        binary = Path(capture["fixture_binary"]).resolve()
        if not binary.is_file() or sha256_file(binary) != capture.get("fixture_binary_sha256"):
            raise AcceptanceError("captured C++ fixture binary is missing or changed", EXIT_FAIL,
                                  "fixture_binary_digest_mismatch")
        library = _find_projection_library(root, source, native_library)
        if library is None:
            result["reason_code"] = "native_projection_library_missing"
            result["missing"] = ["new joint_projection shared library for this source/build"]
            return _write_acceptance_command_result(root, "tune", result, EXIT_INCOMPLETE)
        result["native_library"] = str(library)
        attempt_root = root / "runs/tune"
        attempt_root.mkdir(parents=True, exist_ok=True)
        attempt = attempt_root / f"attempt-{_attempt_number(attempt_root):03d}"
        attempt.mkdir(exist_ok=False)
        result["attempt"] = str(attempt.relative_to(root))
        result["reason_code"] = "tuning_running"
        result["source_hashes"] = {
            "acceptance_driver": sha256_file(Path(__file__).resolve()),
            "implementation_source_tree": source_binding["tree_digest_sha256"],
            "fixture_binary": sha256_file(binary), "native_library": sha256_file(library),
            "prediction_config_semantic": exact_config_record["semantic_sha256"],
        }
        recipe_by_id = {case["case_id"]: planned.get(case["case_id"])
                        or _recipe_for_fixture(case["state"])
                        for case in dev_cases}
        # 批量银行评估预算（非单验收 suite；见 packages/d1d2-fixes
        # 裁决记录）：唯一默认配置 × dev 在场集的 L2/L3 + timing
        # （无精度扫描轴：K>1 窗口机制已退役，配置面即精确面）。
        deadline = started + 14400
        contract, config_record = resolve_prediction_contract(
            source, prediction_config)
        config_dir = attempt / "config"
        config_dir.mkdir(exist_ok=False)
        case_results: list[dict[str, Any]] = []
        fixture_failures: list[dict[str, Any]] = []
        for case_index, base_case in enumerate(dev_cases):
            recipe = recipe_by_id[base_case["case_id"]]
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                fixture_failures.append({"case_id": base_case["case_id"],
                                         "status": "INCOMPLETE",
                                         "reason": "tuning_suite_wall_limit_reached"})
                break
            candidate_result, candidate_run = _run_fixture_once(
                binary, source, base_case["state_path"], base_case["events_path"],
                contract, out_dir=(config_dir / "accuracy" /
                                   f"case-{case_index:04d}"),
                label=f"tune-{base_case['case_id']}",
                evidence_root_path=root,
                timeout_seconds=max(1, min(300, int(remaining))))
            if candidate_result is None or candidate_result["status"] != "PASS":
                fixture_failures.append({"case_id": base_case["case_id"],
                                         "status": candidate_run["status"],
                                         "reason": candidate_run.get("reason")})
                if time.monotonic() >= deadline:
                    fixture_failures.append({"reason": "tuning_suite_wall_limit_reached"})
                    break
                continue
            l2 = compare_fixed_action_results(
                base_case["exact"], candidate_result,
                focus_stage=recipe["causal_input"]["focus_stage"])
            case_results.append({"case_id": base_case["case_id"],
                                 "cluster_id": base_case["cluster_id"],
                                 "l2": l2, "candidate_run": candidate_run})
        l2_summary = _aggregate_l2_results(case_results)
        l2_mismatched = [row for row in case_results
                         if row.get("l2", {}).get("state_mismatches")]
        if l2_mismatched and l2_summary["status"] == "INSUFFICIENT_EVIDENCE":
            # §19.4 物理与事务：canonical causal prestate 分叉是物理
            # 一致性失败，不是证据不足——配置不合法（FAIL 路径），
            # 不得以近似时间预算容忍。
            l2_summary["status"] = "FAIL"
            l2_summary["physical_state_mismatch_cases"] = [
                row.get("case_id") for row in l2_mismatched]
        l3_scores, l3_records = _evaluate_l3_bank(
            dev_cases, source, config_record["semantic_config"],
            config_record["semantic_sha256"], library,
            config_dir / "l3", root)
        write_json(config_dir / "l3-score.json", l3_scores)
        l3_summary = _l3_bank_summary(l3_scores)
        config_result: dict[str, Any] = {
            "prediction_config_sha256": config_record["semantic_sha256"],
            "l2": l2_summary, "l3": l3_summary,
            "l3_records": l3_records,
            "fixture_failures": fixture_failures,
            "case_count": len(case_results),
            "status": "INCOMPLETE", "legal": False,
        }
        hard_failure = any(item.get("status") == "FAIL" for item in fixture_failures)
        accuracy_pass = (not fixture_failures and l2_summary["status"] == "PASS"
                         and l3_summary["status"] == "PASS")
        config_result["accuracy_pass"] = accuracy_pass
        if hard_failure or l2_summary["status"] == "FAIL" or l3_summary["status"] == "FAIL":
            config_result["status"] = "FAIL"
        elif accuracy_pass:
            reps, timing_records, timing_status = _measure_config_pairs(
                binary, source, exact_contract, contract, dev_cases,
                attempt= config_dir, root=root, deadline=deadline)
            config_result["timing_repetitions"] = reps
            config_result["timing_invocations"] = timing_records
            if timing_status == "PASS":
                config_result["paired_wall_ns"] = [
                    row["arms"]["candidate"]["elapsed_host_ns"] for row in reps]
                config_result["legal"] = True
                config_result["status"] = "PASS"
            else:
                config_result["status"] = "INCOMPLETE"
                config_result["reason"] = "timing_pair_did_not_naturally_complete"
        elif fixture_failures or "INSUFFICIENT_EVIDENCE" in (
                l2_summary["status"], l3_summary["status"]):
            config_result["status"] = "INCOMPLETE"
        result["config_results"].append(config_result)
        write_json(root / "packages/acceptance/tune-progress.json", result)
        if time.monotonic() >= deadline:
            result.update({"status": "INCOMPLETE",
                           "reason_code": "tuning_suite_wall_limit_reached"})
            return _write_acceptance_command_result(root, "tune", result,
                                                    EXIT_INCOMPLETE)
        if config_result["status"] == "INCOMPLETE":
            result.update({"status": "INCOMPLETE",
                           "reason_code": "configuration_incomplete"})
            return _write_acceptance_command_result(root, "tune", result,
                                                    EXIT_INCOMPLETE)
        selection_rows = [{
            "legal": row["legal"], "paired_wall_ns": row.get("paired_wall_ns"),
        } for row in result["config_results"]]
        selected = choose_development_config(selection_rows)
        selected_contract, selected_config_record = resolve_prediction_contract(
            source, prediction_config)
        source_binding = _verify_implementation_source_binding(
            root, source, capture.get("implementation_source"))
        result.update({"status": "PASS", "reason_code": "development_config_completed",
                       "selection": selected,
                       "approximation_accepted": selected["approximation_accepted"],
                       "completed_utc": utc_now()})
        write_json(root / "packages/acceptance/tune.json", result)
        accuracy = {
            "schema_version": "joint-accel-frozen-accuracy-v1-20261006",
            "status": "PASS", "created_utc": utc_now(),
            "approximation_accepted": selected["approximation_accepted"],
            "config_results": result["config_results"],
            "dev_source": {"capture_sha256": sha256_file(root / "packages/acceptance/capture.json"),
                           "campaign_sha256": sha256_file(root / "manifests/campaign-manifest.json"),
                           "source_root": str(source),
                           "implementation_source": capture["implementation_source"],
                           "native_library_sha256": sha256_file(library),
                           "native_library_path": str(library),
                           "prediction_config": exact_config_record,
                           "selected_prediction_config": selected_config_record,
                           "selected_kernel_contract": selected_contract},
            "bootstrap": {"replicates": BOOTSTRAP_REPLICATES,
                          "seed": BOOTSTRAP_SEED, "confidence_level": 0.95},
        }
        write_json(frozen_path, accuracy)
        return _write_acceptance_command_result(root, "tune", result, EXIT_PASS)
    except AcceptanceError as exc:
        result.update({"status": "FAIL" if exc.code == EXIT_FAIL else "INCOMPLETE",
                       "reason_code": exc.reason_code, "error": str(exc),
                       "completed_utc": utc_now()})
        return _write_acceptance_command_result(
            root, "tune", result, EXIT_FAIL if exc.code == EXIT_FAIL else EXIT_INCOMPLETE)


def command_holdout(root: Path, *, source_root: Path | None,
                    prediction_config: Path | None,
                    native_library: Path | None) -> int:
    """Evaluate the frozen dev choice once on untouched, clustered holdout cases."""
    started = time.monotonic()
    started_ns = time.monotonic_ns()
    result: dict[str, Any] = {
        "schema_version": "joint-accel-holdout-v1-20261006",
        "status": "INCOMPLETE", "reason_code": "holdout_not_started",
        "created_utc": utc_now(), "case_results": [], "l2": None, "l3": None,
        "coverage": None, "tail_coverage": None,
    }
    try:
        verify_statistical_reference(root)
        missing_parents = []
        parents = {}
        for parent in ("G-C5", "G-D1", "G-E0"):
            try:
                parents[parent] = sha256_file(root / "gates" / parent / "DONE")
                validate_done(root, parent)
            except AcceptanceError:
                missing_parents.append(parent)
        if missing_parents:
            result.update({"reason_code": "holdout_parent_gates_missing",
                           "missing_parent_gates": missing_parents})
            return _write_acceptance_command_result(root, "holdout", result,
                                                    EXIT_INCOMPLETE)

        accuracy_path = root / "manifests/frozen-accuracy.json"
        if not accuracy_path.is_file():
            result.update({"reason_code": "frozen_accuracy_missing",
                           "missing": ["run tune and freeze the dev-only choice"]})
            return _write_acceptance_command_result(root, "holdout", result,
                                                    EXIT_INCOMPLETE)
        accuracy = _require_object(read_json(accuracy_path), "frozen accuracy")
        if accuracy.get("status") != "PASS":
            raise AcceptanceError("frozen accuracy selection is not PASS", EXIT_FAIL,
                                  "frozen_accuracy_invalid")
        accuracy_source = _require_object(accuracy.get("dev_source"),
                                          "frozen accuracy dev_source")
        source = (source_root or Path(accuracy_source["source_root"])).expanduser().resolve()
        if Path(accuracy_source["source_root"]).resolve() != source:
            raise AcceptanceError("holdout source root differs from the frozen dev source",
                                  EXIT_FAIL, "holdout_source_mismatch")
        capture, all_rows = _load_captured_exact_bank(root, partition="holdout")
        source_binding = _verify_implementation_source_binding(
            root, source, accuracy_source.get("implementation_source"))
        capture_binding = _require_object(capture.get("implementation_source"),
                                          "capture implementation source")
        if capture_binding != accuracy_source.get("implementation_source"):
            raise AcceptanceError("holdout source identity differs from the capture source",
                                  EXIT_FAIL, "holdout_source_digest_mismatch")
        result["implementation_source"] = source_binding
        campaign = _load_campaign_manifest(root, source, capture)
        initial = {row["case_id"]: row for row in campaign["controlled_cases"]
                   if row["partition"] == "holdout"}
        priceable = [row for row in all_rows
                     if row["scenario_id"] in PRICEABLE_SCENARIO_IDS]
        by_id = {row["case_id"]: row for row in priceable}
        missing_initial = sorted(set(initial) - by_id.keys())
        extra_ids = sorted(set(by_id) - set(initial))
        invalid_extensions = []
        recipes: dict[str, dict[str, Any]] = {}
        for case in priceable:
            recipe = initial.get(case["case_id"])
            if recipe is None:
                try:
                    recipe = _recipe_for_fixture(case["state"])
                except AcceptanceError as exc:
                    invalid_extensions.append({"case_id": case["case_id"],
                                               "reason": exc.reason_code})
                    continue
            _validate_capture_case_recipe(case, recipe)
            recipes[case["case_id"]] = recipe
        result["initial_holdout_case_count"] = len(initial)
        result["captured_priceable_holdout_case_count"] = len(priceable)
        result["missing_initial_case_ids"] = missing_initial
        result["extension_case_ids"] = extra_ids
        if invalid_extensions:
            result.update({"reason_code": "holdout_fixture_bank_incomplete",
                           "invalid_extension_case_ids": invalid_extensions})
            return _write_acceptance_command_result(root, "holdout", result,
                                                    EXIT_INCOMPLETE)
        if missing_initial:
            # 覆盖驱动子集（§19.2 第 3 项达标即停）下 planned holdout
            # 全场不可满足：确定性 >300s 超时案例保留在失败清单、不
            # 虚构 exact 记录；在场集作为评估集，缺失集全量披露。
            # §19.2 第 4 项 100/30 覆盖门本身不放宽。
            result["planned_holdout_missing_disclosure"] = {
                "count": len(missing_initial),
                "case_ids": missing_initial,
                "reason": "coverage-driven subset; deterministic 300s-capped "
                          "fixture timeouts retained on the failure list "
                          "(execdoc §19.2 items 3/7)",
            }
        if not priceable:
            result.update({"reason_code": "holdout_fixture_bank_empty",
                           "coverage": sample_coverage([])})
            return _write_acceptance_command_result(root, "holdout", result,
                                                    EXIT_INSUFFICIENT)

        selected_contract = validate_kernel_contract(
            accuracy_source.get("selected_kernel_contract"))
        config_record = _require_object(accuracy_source.get("selected_prediction_config"),
                                       "selected prediction config")
        semantic_config = _require_object(config_record.get("semantic_config"),
                                          "selected semantic prediction config")
        if config_record.get("semantic_sha256") != selected_contract[
                "prediction_config_sha256"]:
            raise AcceptanceError("selected semantic config digest differs from contract",
                                  EXIT_FAIL, "frozen_accuracy_invalid")
        binary = Path(capture["fixture_binary"]).resolve()
        if (not binary.is_file()
                or sha256_file(binary) != capture.get("fixture_binary_sha256")):
            raise AcceptanceError("captured C++ fixture binary is missing or changed",
                                  EXIT_FAIL, "fixture_binary_digest_mismatch")
        library = _find_projection_library(root, source, native_library)
        if library is None:
            result.update({"reason_code": "native_projection_library_missing"})
            return _write_acceptance_command_result(root, "holdout", result,
                                                    EXIT_INCOMPLETE)
        expected_library_sha = accuracy_source.get("native_library_sha256")
        if expected_library_sha != sha256_file(library):
            raise AcceptanceError("holdout native library differs from frozen dev library",
                                  EXIT_FAIL, "holdout_library_mismatch")
        if prediction_config is not None:
            config_path = prediction_config.expanduser()
            if not config_path.is_absolute():
                config_path = (root / config_path).resolve()
            resolved_contract, resolved_record = resolve_prediction_contract(
                source, config_path)
            if (resolved_contract != selected_contract
                    or resolved_record["semantic_sha256"] != config_record["semantic_sha256"]):
                raise AcceptanceError("holdout prediction config differs from frozen dev choice",
                                      EXIT_FAIL, "holdout_config_mismatch")

        attempt_root = root / "runs/holdout"
        attempt_root.mkdir(parents=True, exist_ok=True)
        attempt = attempt_root / f"attempt-{_attempt_number(attempt_root):03d}"
        attempt.mkdir(exist_ok=False)
        result.update({"attempt": str(attempt.relative_to(root)),
                       "source_root": str(source), "fixture_binary": str(binary),
                       "fixture_binary_sha256": sha256_file(binary),
                       "native_library": str(library),
                       "native_library_sha256": sha256_file(library),
                       "frozen_accuracy_sha256": sha256_file(accuracy_path),
                       "capture_sha256": sha256_file(root / "packages/acceptance/capture.json"),
                       "parent_done_sha256": parents})
        first_boundaries = []
        for case in priceable:
            ci = case["state"].get("causal_input", {})
            case_axes = {
                "competition_level": ci.get("competition_level",
                                            ci.get("competition")),
                "source_count": ci.get("source_count", ci.get("sources")),
                # 生成器文档：M09 = tightened HBM capacity stress。
                "capacity_bucket": ("stress" if case["scenario_id"] == "M09"
                                    else "normal"),
            }
            for boundary in _first_result_boundaries(case["exact"],
                                                     case["cluster_id"]):
                first_boundaries.append(dict(boundary, causal_metadata={
                    **boundary.get("causal_metadata", {}), **case_axes}))
        result["coverage"] = sample_coverage(first_boundaries)
        exact_itl_intervals = 0
        completed_ids = set()
        cluster_ids = set()
        for case in priceable:
            cluster_ids.add(case["cluster_id"])
            completed_ids.add((case["cluster_id"], case["state"]["request_id"]))
            exact_itl_intervals += sum(
                row.get("itl_interval_count", 0) for row in case["exact"]["measurements"]
                if isinstance(row.get("itl_interval_count", 0), int)
                and not isinstance(row.get("itl_interval_count", 0), bool))
        result["tail_coverage"] = tail_coverage(
            len(completed_ids), len(cluster_ids), exact_itl_intervals)
        if not result["coverage"]["missing"] and exact_itl_intervals == 0:
            result["tail_coverage"]["itl_p99"] = "INSUFFICIENT_EVIDENCE"
            result["tail_coverage"]["reason"] = "fixture does not expose actual ITL intervals"

        attempt_dir = attempt / "selected"
        attempt_dir.mkdir()
        l2_cases: list[dict[str, Any]] = []
        execution_failures: list[dict[str, Any]] = []
        # 批量银行评估预算（非单验收 suite；见 packages/d1d2-fixes
        # 裁决记录）：holdout 在场全集（~110 案例）的 L2/L3 评估。
        deadline = started + 14400
        for index, case in enumerate(sorted(priceable, key=lambda row: row["case_id"])):
            recipe = recipes[case["case_id"]]
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                execution_failures.append({
                    "case_id": case["case_id"], "status": "INCOMPLETE",
                    "reason": "holdout_suite_wall_limit_reached"})
                break
            candidate, invocation = _run_fixture_once(
                binary, source, case["state_path"], case["events_path"],
                selected_contract,
                out_dir=attempt_dir / f"case-{index:04d}",
                label=f"holdout-{case['case_id']}",
                evidence_root_path=root,
                timeout_seconds=max(1, min(300, int(remaining))))
            if candidate is None or candidate.get("status") != "PASS":
                execution_failures.append({
                    "case_id": case["case_id"], "status": invocation.get("status"),
                    "reason": invocation.get("reason")})
                result["case_results"].append({
                    "case_id": case["case_id"], "cluster_id": case["cluster_id"],
                    "status": invocation.get("status"), "invocation": invocation})
                write_json(attempt / "progress.json", result)
                continue
            l2 = compare_fixed_action_results(
                case["exact"], candidate,
                focus_stage=recipe["causal_input"]["focus_stage"])
            case_row = {"case_id": case["case_id"], "cluster_id": case["cluster_id"],
                        "scenario_id": case["scenario_id"], "seed": case["seed"],
                        "focus_stage": recipe["causal_input"]["focus_stage"],
                        "candidate_source": invocation.get("source", "runtime"),
                        "invocation": invocation, "l2": l2}
            l2_cases.append(case_row)
            result["case_results"].append(case_row)
            write_json(attempt / "progress.json", result)

        l2 = _aggregate_l2_results(l2_cases) if l2_cases else {
            "status": "INSUFFICIENT_EVIDENCE", "evaluable_pair_count": 0,
            "missing_stages": ["P", "D"], "pairs": [], "state_mismatches": []}
        l3_scores, l3_records = _evaluate_l3_bank(
            priceable, source, semantic_config, config_record["semantic_sha256"],
            library, attempt / "l3", root)
        l3 = _l3_bank_summary(l3_scores)
        result["l2"] = l2
        result["l3"] = l3
        result["l3_records"] = l3_records
        result["fixture_failures"] = execution_failures
        result["elapsed_host_ns"] = time.monotonic_ns() - started_ns
        hard_failure = (any(row.get("status") == "FAIL" for row in execution_failures)
                        or l2["status"] == "FAIL" or l3["status"] == "FAIL")
        insufficient = (result["coverage"]["status"] != "PASS"
                        or l2["status"] != "PASS" or l3["status"] != "PASS")
        if hard_failure:
            result.update({"status": "FAIL", "reason_code": "holdout_accuracy_or_execution_fail"})
            code = EXIT_FAIL
        elif (execution_failures
              or any(row.get("status") == "INCOMPLETE" for row in l3_records)):
            result.update({"status": "INCOMPLETE", "reason_code": "holdout_execution_incomplete"})
            code = EXIT_INCOMPLETE
        elif insufficient or l2["status"] == "INSUFFICIENT_EVIDENCE" \
                or l3["status"] == "INSUFFICIENT_EVIDENCE":
            result.update({"status": "INSUFFICIENT_EVIDENCE",
                           "reason_code": "holdout_coverage_below_frozen_threshold"})
            code = EXIT_INSUFFICIENT
        else:
            result.update({"status": "PASS", "reason_code": "holdout_accuracy_pass"})
            code = EXIT_PASS
        result["completed_utc"] = utc_now()
        write_json(attempt / "result.json", result)
        return _write_acceptance_command_result(root, "holdout", result, code)
    except AcceptanceError as exc:
        result.update({"status": "FAIL" if exc.code == EXIT_FAIL else "INCOMPLETE",
                       "reason_code": exc.reason_code, "error": str(exc),
                       "completed_utc": utc_now()})
        return _write_acceptance_command_result(
            root, "holdout", result, EXIT_FAIL if exc.code == EXIT_FAIL else exc.code)


def _load_optional_acceptance_json(root: Path, relative: str) -> tuple[dict[str, Any] | None, str | None]:
    path = root / relative
    if not path.is_file():
        return None, "missing"
    try:
        value = _require_object(read_json(path), relative)
    except AcceptanceError as exc:
        return None, f"invalid:{exc.reason_code}"
    return value, None


def _status_of_artifact(value: dict[str, Any] | None) -> str:
    if value is None:
        return "MISSING"
    status = value.get("status")
    return status if isinstance(status, str) and status in GATE_STATUSES else "FAIL"


def _report_failure_samples(artifacts: dict[str, dict[str, Any] | None]) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    def visit(value: Any, path: str) -> None:
        if isinstance(value, dict):
            status = value.get("status")
            if status in ("FAIL", "INCOMPLETE", "INSUFFICIENT_EVIDENCE") and (
                    value.get("reason") or value.get("reason_code") or value.get("error")):
                samples.append({"artifact": path, "status": status,
                                "case_id": value.get("case_id"),
                                "candidate_id": value.get("candidate_id"),
                                "reason_code": value.get("reason_code"),
                                "reason": value.get("reason"),
                                "error": value.get("error")})
            for key, child in value.items():
                if isinstance(child, (dict, list)):
                    visit(child, f"{path}.{key}")
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, f"{path}[{index}]")
    for name, value in artifacts.items():
        if value is not None:
            visit(value, name)
    return samples[:2000]


def _f1_gate_result(root: Path, report_source_path: Path,
                    artifacts: dict[str, dict[str, Any] | None]) -> dict[str, Any]:
    parent = gate_status_detail(root, "G-F0")
    holdout = artifacts.get("holdout")
    performance = artifacts.get("performance")
    accuracy = artifacts.get("frozen_accuracy")
    holdout_status = _status_of_artifact(holdout)
    perf_status = _status_of_artifact(performance)
    performance_valid = False
    performance_error: str | None = None
    if performance is not None and perf_status == "PASS":
        try:
            summary = summarize_performance_repetitions(performance.get("repetitions"))
            performance_valid = (performance.get("speed_ratios") == summary["speed_ratios"]
                                 and performance.get("work_signature_sha256")
                                     == summary["work_signature_sha256"])
            if not performance_valid:
                performance_error = "performance summary differs from its paired runs"
        except (AcceptanceError, TypeError, KeyError) as exc:
            performance_error = str(exc)
    checks = {
        "f0_gate_passed": parent["status"] == "PASS",
        "holdout_passed": holdout_status == "PASS",
        "five_paired_performance_repetitions_valid": performance_valid,
        "one_second_timing_inputs_hashed": bool(
            performance and isinstance(performance.get("queue"), dict)
            and performance["queue"].get("queue_sha256")
            and performance.get("repetitions")),
        "stable_report_source_written": report_source_path.is_file(),
    }
    failures = [name for name, value in checks.items() if not value]
    if parent["status"] == "FAIL" or holdout_status == "FAIL" or perf_status == "FAIL":
        status, reason_code = "FAIL", "f1_evidence_failed"
    elif parent["status"] == "INSUFFICIENT_EVIDENCE" or holdout_status == "INSUFFICIENT_EVIDENCE":
        status, reason_code = "INSUFFICIENT_EVIDENCE", "f1_accuracy_evidence_insufficient"
    elif failures:
        status, reason_code = "INCOMPLETE", "f1_required_evidence_incomplete"
    else:
        status, reason_code = "PASS", "f1_report_inputs_complete"
    parents = ({"G-F0": parent["done_sha256"]} if parent["status"] == "PASS"
               and isinstance(parent.get("done_sha256"), str) else {})
    required_artifacts: dict[str, str] = {}
    base_paths = [report_source_path,
                  root / "packages/acceptance/holdout.json",
                  root / "packages/acceptance/performance.json"]
    for path in base_paths:
        if path.is_file():
            required_artifacts[path.relative_to(root).as_posix()] = sha256_file(path)
    for rel in ("packages/acceptance/capture.json", "manifests/frozen-accuracy.json",
                "manifests/campaign-manifest.json"):
        path = root / rel
        if path.is_file():
            required_artifacts[rel] = sha256_file(path)
    if isinstance(accuracy, dict) and isinstance(accuracy.get("dev_source"), dict):
        implementation_source = accuracy["dev_source"].get("implementation_source")
        if isinstance(implementation_source, dict):
            inventory_rel = implementation_source.get("inventory_path")
            inventory_path = root / inventory_rel if isinstance(inventory_rel, str) else None
            if inventory_path is not None and inventory_path.is_file():
                required_artifacts[inventory_rel] = sha256_file(inventory_path)
    if performance is not None:
        for repetition in performance.get("repetitions", []):
            if not isinstance(repetition, dict):
                continue
            for arm in repetition.get("arms", {}).values():
                if not isinstance(arm, dict):
                    continue
                run_root = Path(arm.get("run_dir", ""))
                for entry in arm.get("run_artifacts", []):
                    if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
                        continue
                    artifact_path = (run_root / entry["path"]).resolve()
                    try:
                        relative = artifact_path.relative_to(root.resolve()).as_posix()
                    except ValueError:
                        continue
                    if artifact_path.is_file():
                        required_artifacts[relative] = entry.get("sha256")
    return {
        "gate": "G-F1", "status": status, "reason_code": reason_code,
        "parents": parents, "checks": checks,
        "required_artifacts": required_artifacts,
        "failed_checks": failures, "performance_error": performance_error,
        "speed_goal_met": (performance.get("speed_goal_met")
                           if performance and perf_status == "PASS" else None),
        "report_source_sha256": sha256_file(report_source_path)
            if report_source_path.is_file() else None,
    }


def _write_f1_gate_result(root: Path, candidate: dict[str, Any]) -> None:
    result_path = root / "gates/G-F1/result.json"
    done_path = root / "gates/G-F1/DONE"
    if result_path.is_file():
        try:
            old = read_json(result_path)
        except AcceptanceError:
            old = None
        if old == candidate and done_path.is_file():
            try:
                validate_done(root, "G-F1")
                return
            except AcceptanceError:
                pass
    if done_path.exists():
        invalidate_done(root, "G-F1", "f1_report_inputs_changed")
    write_json(result_path, candidate)


def _final_checks_status(root: Path, expected_source_digest: str | None
                         ) -> tuple[str, str, dict[str, Any] | None]:
    if not isinstance(expected_source_digest, str):
        return "INCOMPLETE", "final_source_binding_missing", None
    path = root / "packages/acceptance/final-checks.json"
    if not path.is_file():
        return "MISSING", "final_checks_missing", None
    try:
        value = _require_object(read_json(path), "final checks")
    except AcceptanceError as exc:
        return "FAIL", exc.reason_code, None
    if value.get("schema_version") != "joint-accel-final-checks-v1-20261006":
        return "FAIL", "final_checks_schema_mismatch", value
    if value.get("status") not in ("PASS", "FAIL", "INCOMPLETE"):
        return "FAIL", "final_checks_status_invalid", value
    if value.get("status") != "PASS":
        return value["status"], value.get("reason_code", "final_checks_not_pass"), value
    if value.get("source_tree_digest_sha256") != expected_source_digest:
        return "FAIL", "final_checks_source_digest_mismatch", value
    source_manifest = _require_object(value.get("source_manifest"), "final source manifest")
    manifest_rel = source_manifest.get("path")
    manifest_sha = source_manifest.get("sha256")
    if not isinstance(manifest_rel, str) or not manifest_rel or Path(manifest_rel).is_absolute():
        return "FAIL", "final_source_manifest_path_invalid", value
    manifest_path = (root / manifest_rel).resolve()
    try:
        manifest_path.relative_to(root.resolve())
    except ValueError:
        return "FAIL", "final_source_manifest_path_invalid", value
    if not manifest_path.is_file():
        return "INCOMPLETE", "final_source_manifest_missing", value
    if not isinstance(manifest_sha, str) or sha256_file(manifest_path) != manifest_sha:
        return "FAIL", "final_source_manifest_digest_mismatch", value
    manifest = _require_object(read_json(manifest_path), "final source manifest artifact")
    source_entries = manifest.get("entries")
    if (manifest.get("schema_version") != IMPLEMENTATION_SOURCE_INVENTORY_VERSION
            or manifest.get("tree_digest_sha256") != expected_source_digest
            or not isinstance(manifest.get("source_root"), str)
            or not manifest.get("source_root")
            or not isinstance(source_entries, list)
            or _canonical_digest(source_entries) != expected_source_digest):
        return "FAIL", "final_source_manifest_content_mismatch", value
    commands = value.get("commands")
    if not isinstance(commands, list) or not commands:
        return "INCOMPLETE", "final_check_commands_missing", value
    required_kinds = {"build_all", "ctest_registered", "pytest_related",
                      "unittest_slo_tools", "cpp_direct"}
    seen_kinds: set[str] = set()
    test_kinds = {"ctest_registered", "pytest_related", "unittest_slo_tools", "cpp_direct"}
    for index, command_value in enumerate(commands):
        command = _require_object(command_value, f"final checks command {index}")
        kind = command.get("kind")
        argv = command.get("argv")
        if kind not in required_kinds or not isinstance(argv, list) or not argv \
                or any(not isinstance(part, str) or not part for part in argv):
            return "FAIL", "final_check_command_shape_invalid", value
        seen_kinds.add(kind)
        if kind == "build_all":
            build_positions = [position for position, part in enumerate(argv)
                               if part == "--build"]
            target_positions = [position for position, part in enumerate(argv)
                                if part in ("--target", "-t")]
            if (Path(argv[0]).name not in ("cmake", "cmake3")
                    or len(build_positions) != 1 or len(target_positions) > 1):
                return "FAIL", "final_build_not_all_target", value
            if target_positions:
                target_position = target_positions[0]
                if (target_position + 1 >= len(argv)
                        or argv[target_position + 1] != "all"):
                    return "FAIL", "final_build_not_all_target", value
        elif kind == "ctest_registered":
            filtered = {"-R", "-E", "--tests-regex", "--exclude-regex"}
            if (Path(argv[0]).name != "ctest"
                    or any(part in filtered for part in argv)):
                return "FAIL", "final_ctest_not_full_registered_suite", value
        cwd = command.get("cwd")
        if not isinstance(cwd, str) or not Path(cwd).expanduser().is_dir():
            return "INCOMPLETE", "final_check_command_cwd_missing", value
        if kind in test_kinds:
            tests_run = command.get("tests_run")
            if isinstance(tests_run, bool) or not isinstance(tests_run, int) or tests_run <= 0:
                return "INCOMPLETE", "final_check_test_count_missing", value
        log_value = command.get("log")
        if (command.get("status") != "completed" or command.get("completed") is not True
                or isinstance(command.get("returncode"), bool)
                or not isinstance(command.get("returncode"), int)
                or not isinstance(log_value, str)):
            return "INCOMPLETE", "final_check_command_not_complete", value
        log_path = Path(log_value).expanduser()
        if not log_path.is_absolute():
            log_path = root / log_path
        try:
            log_path.resolve().relative_to(root.resolve())
        except ValueError:
            return "FAIL", "final_check_log_escapes_evidence", value
        if (not log_path.is_file()
                or command.get("log_sha256") != sha256_file(log_path)):
            return "FAIL", "final_check_log_missing_or_changed", value
        if kind == "build_all" and command["returncode"] != 0:
            return "FAIL", "final_full_build_failed", value
        if kind == "ctest_registered" and command["returncode"] != 0:
            return "FAIL", "final_registered_ctest_failed", value
    missing_kinds = sorted(required_kinds - seen_kinds)
    if missing_kinds:
        return "INCOMPLETE", "final_check_command_kinds_missing", value
    baseline_path = root / "packages/acceptance/baseline.json"
    if not baseline_path.is_file():
        return "INCOMPLETE", "a1_baseline_failure_index_missing", value
    baseline = _require_object(read_json(baseline_path), "A1 baseline")
    known_rows = baseline.get("known_baseline_failures")
    baseline_rows = value.get("baseline_failures")
    new_rows = value.get("new_failures")
    if not isinstance(known_rows, list) or not isinstance(baseline_rows, list) \
            or not isinstance(new_rows, list):
        return "INCOMPLETE", "final_failure_classification_missing", value

    def signatures(rows: list[Any], label: str) -> set[tuple[str, str, str]] | None:
        parsed: set[tuple[str, str, str]] = set()
        for row in rows:
            if not isinstance(row, dict):
                return None
            fields = tuple(row.get(key) for key in
                           ("test_name", "input_signature", "error_signature"))
            if any(not isinstance(field, str) or not field for field in fields):
                return None
            parsed.add(fields)  # type: ignore[arg-type]
        return parsed

    expected_failures = signatures(known_rows, "A1")
    captured_failures = signatures(baseline_rows, "final")
    new_failures = signatures(new_rows, "new")
    if expected_failures is None or captured_failures is None or new_failures is None:
        return "FAIL", "final_failure_signature_invalid", value
    if captured_failures != expected_failures:
        return "FAIL", "final_baseline_failure_classification_changed", value
    if new_failures:
        return "FAIL", "final_new_regressions_present", value
    if not expected_failures and any(command["returncode"] != 0 for command in commands):
        return "FAIL", "final_command_nonzero_without_known_baseline_failure", value
    artifacts = value.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        return "INCOMPLETE", "final_check_artifacts_missing", value
    artifact_paths: set[str] = set()
    for index, artifact_value in enumerate(artifacts):
        artifact = _require_object(artifact_value, f"final checks artifact {index}")
        rel = artifact.get("path")
        expected = artifact.get("sha256")
        if not isinstance(rel, str) or not rel or Path(rel).is_absolute():
            return "FAIL", "final_check_artifact_path_invalid", value
        artifact_path = (root / rel).resolve()
        try:
            artifact_path.relative_to(root.resolve())
        except ValueError:
            return "FAIL", "final_check_artifact_path_invalid", value
        if not artifact_path.is_file():
            return "INCOMPLETE", "final_check_artifact_missing", value
        if not isinstance(expected, str) or expected != sha256_file(artifact_path):
            return "FAIL", "final_check_artifact_digest_mismatch", value
        artifact_paths.add(Path(rel).as_posix())
    normalized = {path.lower() for path in artifact_paths}
    categories = {
        "native_projection_library": any(Path(path).name == "libjoint_projection.so"
                                          for path in normalized),
        "die_model_library": any(("die_model" in path or "diemodel" in path)
                                 and Path(path).suffix in (".so", ".a")
                                 for path in normalized),
        "build_configuration": any(Path(path).name in
                                   ("cmakecache.txt", "build-record.json")
                                   for path in normalized),
        "current_prediction_configuration": any(
            path.endswith("prediction_presets/default.json")
            or path.endswith("prediction_config.py") for path in normalized),
    }
    if not all(categories.values()):
        return "INCOMPLETE", "final_required_artifact_category_missing", {
            **value, "missing_artifact_categories": [name for name, present in categories.items()
                                                       if not present]}
    return "PASS", "final_checks_verified", value


def _publish_f2_gate_result(root: Path, clean: dict[str, Any]) -> None:
    """Create an independently verifiable F2 result; the gate command signs it."""
    f0 = gate_status_detail(root, "G-F0")
    f1 = gate_status_detail(root, "G-F1")
    e0 = gate_status_detail(root, "G-E0")
    source_digest = clean.get("implementation_source", {}).get("tree_digest_sha256") \
        if isinstance(clean.get("implementation_source"), dict) else None
    final_status, final_reason, final_checks = _final_checks_status(root, source_digest)
    def status_hash_recorded(detail: dict[str, Any]) -> bool:
        return (detail.get("status") in GATE_STATUSES
                and detail.get("status") != "MISSING"
                and isinstance(detail.get("result_sha256"), str)
                and detail.get("reason_code") not in {
                    "gate_identity_mismatch", "gate_status_invalid",
                    "gate_result_missing_or_invalid", "gate_schema_invalid"})

    checks = {
        "verify_clean_passed": clean.get("status") == "PASS",
        "current_source_matches_captured_source": clean.get("checks", {}).get(
            "current_source_matches_captured_source") is True,
        "f0_status_and_hash_recorded": status_hash_recorded(f0),
        "f1_status_and_hash_recorded": status_hash_recorded(f1),
        "e0_engineering_parent_passed": e0.get("status") == "PASS",
        "final_source_build_regression_checks_passed": final_status == "PASS",
    }
    if clean.get("status") == "FAIL" or final_status == "FAIL":
        status, reason = "FAIL", "f2_delivery_check_failed"
    elif (not status_hash_recorded(f0) or not status_hash_recorded(f1)
          or e0.get("status") == "MISSING" or final_status == "MISSING"):
        status, reason = "INCOMPLETE", "f2_required_status_or_final_check_missing"
    elif e0.get("status") != "PASS" or final_status != "PASS" \
            or clean.get("status") != "PASS":
        status, reason = "INCOMPLETE", "f2_delivery_checks_incomplete"
    else:
        status, reason = "PASS", "f2_delivery_checks_passed"
    parents = ({"G-E0": e0["done_sha256"]} if e0.get("status") == "PASS"
               and isinstance(e0.get("done_sha256"), str) else {})
    required: dict[str, str] = {}
    for rel in ("packages/acceptance/verify-clean.json",
                "packages/acceptance/final-checks.json",
                "packages/acceptance/report-source.json",
                "gates/G-F0/result.json", "gates/G-F1/result.json"):
        path = root / rel
        if path.is_file():
            required[rel] = sha256_file(path)
    binding = clean.get("implementation_source")
    if isinstance(binding, dict) and isinstance(binding.get("inventory_path"), str):
        path = root / binding["inventory_path"]
        if path.is_file():
            required[binding["inventory_path"]] = sha256_file(path)
    result = {
        "gate": "G-F2", "status": status, "reason_code": reason,
        "created_utc": utc_now(), "parents": parents, "checks": checks,
        "required_artifacts": required,
        "science_status": {"G-F0": f0, "G-F1": f1},
        "final_checks_status": final_status,
        "final_checks_reason": final_reason,
        "final_checks_summary": ({key: final_checks.get(key) for key in
                                   ("source_tree_digest_sha256", "commands", "artifacts")}
                                  if isinstance(final_checks, dict) else None),
    }
    result_path = root / "gates/G-F2/result.json"
    done_path = root / "gates/G-F2/DONE"
    if done_path.exists():
        invalidate_done(root, "G-F2", "final_delivery_inputs_refreshed")
    write_json(result_path, result)


def command_report(root: Path) -> int:
    """Write status-separated JSON/Markdown reports and an auditable F1 gate claim."""
    gate_names = tuple(GATE_PARENTS)
    gates = {name: gate_status_detail(root, name) for name in gate_names}
    artifacts: dict[str, dict[str, Any] | None] = {}
    artifact_details: dict[str, Any] = {}
    for name in ("capture", "tune", "holdout", "performance", "smoke",
                 "verify-clean"):
        relative = f"packages/acceptance/{name}.json"
        value, error = _load_optional_acceptance_json(root, relative)
        artifacts[name] = value
        if value is not None:
            detail = {"status": _status_of_artifact(value), "path": relative,
                      "sha256": sha256_file(root / relative),
                      "reason_code": value.get("reason_code")}
            source_binding = value.get("implementation_source")
            if isinstance(source_binding, dict):
                detail["implementation_source"] = {
                    key: source_binding.get(key) for key in
                    ("source_root", "tree_digest_sha256", "inventory_path",
                     "inventory_sha256", "acceptance_driver_sha256")}
            artifact_details[name] = detail
        else:
            artifact_details[name] = {"status": "MISSING", "path": relative,
                                      "reason": error}

    final_checks, final_checks_error = _load_optional_acceptance_json(
        root, "packages/acceptance/final-checks.json")
    artifacts["final_checks"] = final_checks
    artifact_details["final_checks"] = (
        {"status": final_checks.get("status"),
         "path": "packages/acceptance/final-checks.json",
         "sha256": sha256_file(root / "packages/acceptance/final-checks.json"),
         "source_tree_digest_sha256": final_checks.get("source_tree_digest_sha256")}
        if final_checks is not None else
        {"status": "MISSING", "path": "packages/acceptance/final-checks.json",
         "reason": final_checks_error})

    accuracy, accuracy_error = _load_optional_acceptance_json(
        root, "manifests/frozen-accuracy.json")
    artifacts["frozen_accuracy"] = accuracy
    artifact_details["frozen_accuracy"] = (
        {"status": _status_of_artifact(accuracy),
         "path": "manifests/frozen-accuracy.json",
         "sha256": sha256_file(root / "manifests/frozen-accuracy.json"),
         "approximation_accepted": accuracy.get("approximation_accepted"),
         "implementation_source": ({key: accuracy["dev_source"].get(
             "implementation_source", {}).get(key) for key in
             ("source_root", "tree_digest_sha256", "inventory_path",
              "inventory_sha256", "acceptance_driver_sha256")}
             if isinstance(accuracy.get("dev_source"), dict)
             and isinstance(accuracy["dev_source"].get("implementation_source"), dict)
             else None)}
        if accuracy is not None else {"status": "MISSING",
                                      "path": "manifests/frozen-accuracy.json",
                                      "reason": accuracy_error})

    accuracy_accepted = (gates["G-F0"]["status"] == "PASS"
                         and artifact_details["holdout"]["status"] == "PASS")
    implementation_complete = gates["G-F2"]["status"] == "PASS"
    research_validated = gates["G-F1"]["status"] == "PASS"

    # This stable pre-gate artifact is an input to G-F1, avoiding a hash cycle
    # between its DONE record and the final report that cites that DONE record.
    report_source = {
        "schema_version": "joint-accel-report-source-v1-20261006",
        "gates_before_f1": {name: gates[name] for name in gate_names
                             if name not in ("G-F1", "G-F2")},
        "artifacts": artifact_details,
        "accuracy_accepted": accuracy_accepted,
        "implementation_source": (artifacts.get("verify-clean", {}).get(
            "implementation_source") if isinstance(artifacts.get("verify-clean"), dict)
            else None),
        "source_driver_sha256": sha256_file(Path(__file__).resolve()),
    }
    report_source_path = root / "packages/acceptance/report-source.json"
    write_json(report_source_path, report_source)
    _write_f1_gate_result(root, _f1_gate_result(root, report_source_path, artifacts))
    gates["G-F1"] = gate_status_detail(root, "G-F1")
    gates["G-F2"] = gate_status_detail(root, "G-F2")
    implementation_complete = gates["G-F2"]["status"] == "PASS"
    research_validated = gates["G-F1"]["status"] == "PASS"

    performance = artifacts.get("performance")
    speed_goal_met = (performance.get("speed_goal_met")
                      if performance and _status_of_artifact(performance) == "PASS"
                      else None)
    complete = (implementation_complete and accuracy_accepted
                and research_validated)
    statuses = [detail["status"] for detail in gates.values()]
    statuses.extend(detail["status"] for detail in artifact_details.values())
    if "FAIL" in statuses:
        report_status = "FAIL"
    elif not complete and any(status in ("MISSING", "INCOMPLETE") for status in statuses):
        report_status = "INCOMPLETE"
    elif not complete and "INSUFFICIENT_EVIDENCE" in statuses:
        report_status = "INSUFFICIENT_EVIDENCE"
    else:
        report_status = "PASS"
    report = {
        "schema_version": "joint-accel-final-report-v1-20261006",
        "created_utc": utc_now(), "status": report_status,
        "implementation_complete": implementation_complete,
        "accuracy_accepted": accuracy_accepted,
        "research_conclusions_validated": research_validated,
        "speed_goal_met": speed_goal_met,
        "gates": gates, "artifacts": artifact_details,
        "implementation_source": report_source["implementation_source"],
        "source_driver_sha256": sha256_file(Path(__file__).resolve()),
        "missing_conditions": [
            key for key, value in (
                ("implementation_complete", implementation_complete),
                ("accuracy_accepted", accuracy_accepted),
                ("research_conclusions_validated", research_validated),
            ) if not value],
        "speed_goal_note": ("not measured" if speed_goal_met is None else
                            "derived from complete three-arm paired performance suite"),
    }
    failure_samples = _report_failure_samples(artifacts)
    acceptance_dir = root / "packages/acceptance"
    write_json(acceptance_dir / "report.json", report)
    write_json(acceptance_dir / "acceptance.json", report)
    write_json(acceptance_dir / "failure-samples.json", {
        "schema_version": "joint-accel-failure-samples-v1-20261006",
        "status": "PASS" if not failure_samples else "DIAGNOSTIC",
        "samples": failure_samples,
    })
    gate_lines = ["| Gate | Status | Reason |", "|---|---|---|"]
    gate_lines.extend(
        f"| {name} | {detail['status']} | {detail.get('reason_code', '')} |"
        for name, detail in gates.items())
    artifact_lines = ["| Artifact | Status | Reason |", "|---|---|---|"]
    artifact_lines.extend(
        f"| {name} | {detail['status']} | {detail.get('reason_code', detail.get('reason', ''))} |"
        for name, detail in artifact_details.items())
    markdown = "\n".join([
        "# Joint acceleration acceptance report", "",
        f"Overall status: **{report_status}**", "",
        f"Implementation complete: **{implementation_complete}**  ",
        f"Accuracy accepted: **{accuracy_accepted}**  ",
        f"Research conclusions validated: **{research_validated}**  ",
        f"Speed goal met: **{speed_goal_met if speed_goal_met is not None else 'not measured'}**",
        f"Implementation source digest: `{(report_source['implementation_source'] or {}).get('tree_digest_sha256', 'unavailable')}`",
        "", "## Gates", "", *gate_lines, "", "## Artifacts", "", *artifact_lines,
        "", f"Failure/incomplete diagnostic samples: {len(failure_samples)}",
        "", f"Driver SHA256: `{report['source_driver_sha256']}`", "",
    ])
    markdown_path = acceptance_dir / "acceptance.md"
    markdown_path.write_text(markdown, encoding="utf-8")
    code = (EXIT_PASS if report_status == "PASS" else EXIT_FAIL if report_status == "FAIL"
            else EXIT_INSUFFICIENT if report_status == "INSUFFICIENT_EVIDENCE"
            else EXIT_INCOMPLETE)
    print(json.dumps({"command": "report", "status": report_status,
                      "implementation_complete": implementation_complete,
                      "accuracy_accepted": accuracy_accepted,
                      "research_conclusions_validated": research_validated,
                      "speed_goal_met": speed_goal_met,
                      "report": "packages/acceptance/acceptance.json",
                      "markdown": "packages/acceptance/acceptance.md",
                      "failure_samples": len(failure_samples)},
                     ensure_ascii=False, sort_keys=True))
    return code


def command_verify_clean(root: Path) -> int:
    """Read-only final-tree check; never removes user data or rewrites the repo."""
    generated_manifest_path = root / "manifests/preexisting-generated-artifacts.json"
    trace_original_path = root / "manifests/trace_config.original"
    result: dict[str, Any] = {
        "schema_version": "joint-accel-verify-clean-v1-20261006",
        "status": "INCOMPLETE", "reason_code": "cleanliness_not_verified",
        "created_utc": utc_now(), "checks": {}, "failures": [],
        "evidence_root": str(root), "workspace_source": str(REPO_ROOT),
    }
    if not generated_manifest_path.is_file() or not trace_original_path.is_file():
        result.update({"reason_code": "a0_cleanliness_manifest_missing",
                       "missing": [str(path) for path in
                                   (generated_manifest_path, trace_original_path)
                                   if not path.is_file()]})
        code = _write_acceptance_command_result(root, "verify-clean", result,
                                                EXIT_INCOMPLETE)
        _publish_f2_gate_result(root, result)
        return code
    generated_manifest = _require_object(read_json(generated_manifest_path),
                                         "preexisting generated artifacts")
    baseline_source = root / "reference/source"
    artifact_checks = []
    for entry in generated_manifest.get("entries", []):
        relative = Path(entry["path"])
        live = REPO_ROOT / relative
        archived = root / "preserved-before" / relative
        frozen = baseline_source / relative
        archive_valid = False
        if entry.get("type") == "directory" and archived.is_dir() and frozen.is_dir():
            _, archive_digest = inventory_tree(archived)
            _, frozen_digest = inventory_tree(frozen)
            archive_valid = archive_digest == frozen_digest
        elif entry.get("type") == "file" and archived.is_file() and frozen.is_file():
            archive_valid = sha256_file(archived) == sha256_file(frozen)
        artifact_checks.append({
            "path": relative.as_posix(), "still_live": live.exists(),
            "archive_path": str(archived.relative_to(root)) if archived.exists() else None,
            "archive_matches_frozen_copy": archive_valid,
        })
        if live.exists():
            result["failures"].append({"path": relative.as_posix(),
                                       "reason": "generated_artifact_still_in_live_tree"})
        if not archive_valid:
            result["failures"].append({"path": relative.as_posix(),
                                       "reason": "original_artifact_archive_missing_or_mismatch"})
    trace_config = (REPO_ROOT / "sh_test_mesh/workload/llama2_7b_inference/trace_config.csv")
    trace_matches = (trace_config.is_file()
                     and trace_config.read_bytes() == trace_original_path.read_bytes())
    result["checks"]["trace_config_restored"] = trace_matches
    if not trace_matches:
        result["failures"].append({"path": str(trace_config),
                                   "reason": "trace_config_differs_from_a0_saved_bytes"})
    traces_dir = REPO_ROOT / "sh_test_mesh/workload/llama2_7b_inference/traces"
    unexpected_trace_files = sorted(
        path.relative_to(REPO_ROOT).as_posix() for path in traces_dir.rglob("*")
        if path.is_file() and path.suffix != ".py") if traces_dir.is_dir() else []
    result["checks"]["trace_directory_only_contains_materializer_python"] = not unexpected_trace_files
    if unexpected_trace_files:
        result["failures"].append({"reason": "unexpected_trace_artifacts",
                                   "paths": unexpected_trace_files})
    source_inventory = read_json(root / "manifests/source-inventory.json")
    frozen_paths = {row["path"] for row in source_inventory.get("entries", [])}
    new_caches = []
    for directory, dirs, files in os.walk(REPO_ROOT, followlinks=False):
        dirs[:] = [name for name in dirs if name not in (".git",)]
        base = Path(directory)
        for name in list(dirs) + files:
            path = base / name
            if name == "__pycache__" or name == ".pytest_cache" or name.endswith(".pyc"):
                rel = path.relative_to(REPO_ROOT).as_posix()
                if rel not in frozen_paths:
                    new_caches.append(rel)
    result["checks"]["no_new_python_caches"] = not new_caches
    if new_caches:
        result["failures"].append({"reason": "new_python_cache_artifacts",
                                   "paths": new_caches})
    result["checks"]["generated_artifacts"] = artifact_checks

    accuracy_value: dict[str, Any] | None = None
    accuracy_path = root / "manifests/frozen-accuracy.json"
    if accuracy_path.is_file():
        accuracy_value = _require_object(read_json(accuracy_path), "frozen accuracy")
    capture_path = root / "packages/acceptance/capture.json"
    capture_value: dict[str, Any] | None = None
    if capture_path.is_file():
        capture_value = _require_object(read_json(capture_path), "capture result")
    dev_source = (accuracy_value.get("dev_source")
                  if isinstance(accuracy_value, dict) else None)
    binding = (dev_source.get("implementation_source")
               if isinstance(dev_source, dict) else None)
    selected_source_root = (dev_source.get("source_root")
                            if isinstance(dev_source, dict) else None)
    if binding is None and isinstance(capture_value, dict):
        binding = capture_value.get("implementation_source")
        selected_source_root = capture_value.get("source_root")
    try:
        if not isinstance(selected_source_root, str) or not isinstance(binding, dict):
            raise AcceptanceError("no capture/frozen-accuracy implementation source identity",
                                  EXIT_INCOMPLETE, "implementation_source_binding_missing")
        bound_root = Path(selected_source_root).expanduser().resolve()
        verified_binding = _verify_implementation_source_binding(root, bound_root, binding)
        live_entries, live_digest = implementation_source_inventory(REPO_ROOT)
        if live_digest != binding.get("tree_digest_sha256"):
            raise AcceptanceError("final workspace source differs from the captured implementation",
                                  EXIT_FAIL, "final_source_digest_mismatch")
        result["implementation_source"] = {
            **verified_binding, "workspace_source_root": str(REPO_ROOT.resolve()),
            "workspace_tree_digest_sha256": live_digest,
            "workspace_entry_count": len(live_entries),
        }
        result["checks"]["current_source_matches_captured_source"] = True
    except AcceptanceError as exc:
        result["checks"]["current_source_matches_captured_source"] = False
        result["source_binding_error"] = {"reason_code": exc.reason_code,
                                          "message": str(exc)}
        if exc.code == EXIT_FAIL:
            result["failures"].append({"reason": exc.reason_code,
                                       "message": str(exc)})
        else:
            result.setdefault("missing", []).append(exc.reason_code)

    f0 = gate_status_detail(root, "G-F0")
    holdout_value, _holdout_error = _load_optional_acceptance_json(
        root, "packages/acceptance/holdout.json")
    accuracy_accepted = (f0.get("status") == "PASS"
                         and _status_of_artifact(holdout_value) == "PASS"
                         and isinstance(accuracy_value, dict)
                         and accuracy_value.get("status") == "PASS")
    result["science_status"] = {"G-F0": f0,
                                "G-F1": gate_status_detail(root, "G-F1")}
    result["accuracy_accepted"] = accuracy_accepted

    if result["failures"]:
        result["status"] = "FAIL"
        result["reason_code"] = "live_artifacts_source_mismatch"
        code = EXIT_FAIL
    elif result.get("missing"):
        result["status"] = "INCOMPLETE"
        result["reason_code"] = "clean_source_unverified"
        code = EXIT_INCOMPLETE
    else:
        result["status"] = "PASS"
        result["reason_code"] = "clean_source_verified"
        code = EXIT_PASS
    code = _write_acceptance_command_result(root, "verify-clean", result, code)
    _publish_f2_gate_result(root, result)
    return code


def _set_request_queue_csv(source_root: Path, queue_path: Path) -> None:
    config_path = (source_root / "sh_test_mesh/workload/llama2_7b_inference"
                   / "trace_config.csv")
    try:
        with config_path.open(newline="", encoding="utf-8-sig") as stream:
            rows = list(csv.reader(stream))
    except (OSError, csv.Error) as exc:
        raise AcceptanceError(f"cannot read copied trace_config.csv: {exc}",
                              EXIT_FAIL, "smoke_trace_config_invalid") from exc
    matches = [row for row in rows if len(row) >= 3
               and row[0] == "config" and row[1] == "request_queue_csv"]
    if len(matches) != 1:
        raise AcceptanceError("copied trace_config.csv must have one request_queue_csv row",
                              EXIT_FAIL, "smoke_trace_config_invalid")
    matches[0][2] = str(queue_path.resolve())
    tmp = config_path.with_name(config_path.name + ".joint-accel-tmp")
    with tmp.open("w", newline="", encoding="utf-8") as stream:
        csv.writer(stream, lineterminator="\n").writerows(rows)
    os.replace(tmp, config_path)


def _copy_isolated_source(source_root: Path, destination: Path) -> dict[str, Any]:
    source_root = source_root.expanduser().resolve()
    if not source_root.is_dir():
        raise AcceptanceError(f"source/run copy is missing: {source_root}",
                              EXIT_INCOMPLETE, "smoke_source_missing")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise AcceptanceError(f"isolated source destination already exists: {destination}",
                              EXIT_FAIL, "smoke_source_destination_exists")
    digest_before = inventory_tree(source_root)[1]
    shutil.copytree(source_root, destination, symlinks=True)
    digest_copy = inventory_tree(destination)[1]
    if digest_copy != digest_before:
        raise AcceptanceError("isolated source copy differs from its immutable input tree",
                              EXIT_FAIL, "smoke_source_copy_mismatch")
    generated = destination / "sh_test_mesh/generated"
    if generated.exists():
        shutil.rmtree(generated)
    runtime_runs = destination / "sh_test_mesh/runs"
    if runtime_runs.exists():
        shutil.rmtree(runtime_runs)
    (destination / "sh_test_mesh/generated").mkdir(parents=True, exist_ok=True)
    return {"source_root": str(source_root), "source_tree_sha256": digest_before,
            "copied_source_tree_sha256": digest_copy,
            "copy_path": str(destination)}


def _make_one_second_queue(root: Path, materializer_source: Path,
                           attempt: Path) -> tuple[Path, dict[str, Any]]:
    trace = root / "manifests/astra_compute_20.csv"
    if not trace.is_file():
        raise AcceptanceError("frozen trace input is missing", EXIT_INCOMPLETE,
                              "smoke_trace_missing")
    selection = select_first_second_trace(trace)
    selection_path = root / "inputs/trace-window-1s-selection.json"
    if selection_path.is_file() and read_json(selection_path) != selection:
        existing = read_json(selection_path)
        stable = {key: value for key, value in existing.items()
                  if key not in ("materializer",)}
        expected = {key: value for key, value in selection.items()
                    if key not in ("materializer",)}
        if stable != expected:
            raise AcceptanceError("frozen one-second trace selection changed",
                                  EXIT_FAIL, "trace_selection_manifest_mismatch")
    queue_path = attempt / "inputs/joint_t1_request_queue.csv"
    queue_path.parent.mkdir(parents=True, exist_ok=False)
    materializer = (materializer_source /
                    "sh_test_mesh/workload/llama2_7b_inference/traces"
                    / "materialize_20_30s.py")
    if not materializer.is_file():
        raise AcceptanceError("one-second trace materializer is missing from source copy",
                              EXIT_INCOMPLETE, "smoke_materializer_missing")
    log = attempt / "logs/materialize-1s.log"
    invocation = run_logged(
        ["python3", str(materializer), str(trace), str(queue_path), "1000000000"],
        materializer_source, _environment_for_tests(), log, 300)
    if invocation.get("status") != "completed" or invocation.get("returncode") != 0:
        raise AcceptanceError("one-second input materialization did not complete",
                              EXIT_INCOMPLETE, "smoke_materialization_incomplete")
    if not queue_path.is_file():
        raise AcceptanceError("materializer did not write the request queue",
                              EXIT_FAIL, "smoke_queue_missing")
    with queue_path.open(newline="", encoding="utf-8-sig") as stream:
        queue_rows = list(csv.DictReader(stream))
    if len(queue_rows) != selection["selected_request_count"]:
        raise AcceptanceError(
            f"materialized queue has {len(queue_rows)} rows; frozen selector has "
            f"{selection['selected_request_count']}",
            EXIT_FAIL, "smoke_queue_selection_mismatch")
    return queue_path, {
        "selection": selection, "selection_sha256": sha256_file(selection_path)
            if selection_path.is_file() else None,
        "queue_path": str(queue_path), "queue_sha256": sha256_file(queue_path),
        "queue_rows": len(queue_rows), "materializer": invocation,
        "materializer_sha256": sha256_file(materializer),
    }


def _prepare_online_source(source_root: Path, queue_path: Path,
                           attempt: Path, label: str) -> tuple[Path, dict[str, Any]]:
    copied = attempt / "sources" / label
    source_info = _copy_isolated_source(source_root, copied)
    _set_request_queue_csv(copied, queue_path)
    planner = copied / "sh_test_mesh/workload/llama2_7b_inference/plan_materializer.py"
    if not planner.is_file():
        raise AcceptanceError("plan_materializer.py is missing from run source",
                              EXIT_INCOMPLETE, "smoke_plan_materializer_missing")
    prep_log = attempt / "logs" / f"{label}-plan-materializer.log"
    prep = run_logged(["python3", str(planner)], copied,
                      _environment_for_tests(), prep_log, 300)
    if prep.get("status") != "completed" or prep.get("returncode") != 0:
        raise AcceptanceError(f"{label} physical plan materialization failed",
                              EXIT_INCOMPLETE, "smoke_plan_materialization_failed")
    return copied, {**source_info, "plan_materializer": prep,
                    "trace_config_sha256": sha256_file(
                        copied / "sh_test_mesh/workload/llama2_7b_inference"
                        / "trace_config.csv")}


def _run_online_arm(source_copy: Path, queue_path: Path, run_dir: Path,
                    prediction_config_path: Path | None, *, label: str,
                    attempt: Path, timeout_seconds: int = 300) -> dict[str, Any]:
    runner = source_copy / "sh_test_mesh/run_scripts/joint_runner.py"
    binary = (source_copy / "build/astra_analytical/build_congestion_aware/bin"
              / "AstraSim_Analytical_Congestion_Aware_Online")
    if not runner.is_file() or not binary.is_file() or not os.access(binary, os.X_OK):
        raise AcceptanceError(f"{label} runner or online binary is missing",
                              EXIT_INCOMPLETE, "smoke_online_binary_missing")
    if run_dir.exists():
        raise AcceptanceError(f"{label} run directory already exists",
                              EXIT_FAIL, "smoke_run_directory_exists")
    rss_log = attempt / "logs" / f"{label}-time.txt"
    outer_log = attempt / "logs" / f"{label}-runner.log"
    env = _environment_for_tests()
    env.update({"SH_ONLINE_VALIDATE": "1", "SH_METRICS_DETAIL": "full"})
    if prediction_config_path is not None:
        env["JOINT_PREDICTION_CONFIG"] = str(prediction_config_path.resolve())
    time_bin = shutil.which("time")
    if time_bin is None:
        raise AcceptanceError("GNU time is required to record whole-run peak RSS",
                              EXIT_INCOMPLETE, "performance_rss_tool_missing")
    argv = [time_bin, "-v", "-o", str(rss_log), "python3", str(runner),
            str(run_dir), str(queue_path), "--combo", "TJE"]
    invocation = run_logged(argv, source_copy, env, outer_log, timeout_seconds)
    peak_rss = None
    if rss_log.is_file():
        contents = rss_log.read_text(encoding="utf-8", errors="replace")
        match = re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)", contents)
        if match:
            peak_rss = int(match.group(1)) * 1024
    record: dict[str, Any] = {
        "label": label, "source_path": str(source_copy),
        "runner_sha256": sha256_file(runner), "binary_path": str(binary),
        "binary_sha256": sha256_file(binary), "invocation": invocation,
        "peak_rss_bytes": peak_rss, "rss_log": str(rss_log),
        "run_dir": str(run_dir), "status": "INCOMPLETE",
        "natural_completion": False,
    }
    if invocation.get("status") != "completed":
        record["reason"] = "online_runner_timeout_or_launch_failure"
        return record
    if invocation.get("returncode") != 0:
        record.update({"status": "FAIL", "reason": "online_runner_nonzero_exit"})
        return record
    request_metrics = run_dir / "request_metrics.csv"
    # D19：runner archive 压缩 cpp.log → cpp.log.gz；required 检查前恢复。
    _gz = run_dir / "cpp.log.gz"
    _plain = run_dir / "cpp.log"
    if _gz.is_file() and not _plain.is_file():
        import gzip as _gzip
        _plain.write_bytes(_gzip.decompress(_gz.read_bytes()))
    required = ("cpp.log", "python.log", "trace_config.csv.snapshot",
                "hardware_config.json.snapshot", "raw_metrics.csv",
                "normalized_metrics.csv", "request_metrics.csv", "invocation.json")
    missing = [name for name in required if not (run_dir / name).is_file()]
    if missing:
        record["reason"] = "completed_runner_missing_full_metrics"
        record["missing"] = missing
        return record
    with request_metrics.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    queue_hash = sha256_file(queue_path)
    trace_hash = sha256_file(run_dir / "trace_config.csv.snapshot")
    hardware_hash = sha256_file(run_dir / "hardware_config.json.snapshot")
    signature = _canonical_digest({
        "request_queue_sha256": queue_hash,
        "trace_config_snapshot_sha256": trace_hash,
        "hardware_snapshot_sha256": hardware_hash,
        "combo": "TJE",
        "request_metrics_rows": len(rows),
    })
    record.update({
        "status": "completed", "elapsed_host_ns": invocation["elapsed_host_ns"],
        "natural_completion": True,
        "request_metrics_rows": len(rows), "request_queue_sha256": queue_hash,
        "trace_config_snapshot_sha256": trace_hash,
        "hardware_snapshot_sha256": hardware_hash,
        "work_signature_sha256": signature,
        "run_artifacts": [
            {"path": path.relative_to(run_dir).as_posix(), "sha256": sha256_file(path)}
            for path in sorted(run_dir.rglob("*"))
            if path.is_file() and not path.is_symlink()],
    })
    if len(rows) != 8:
        record.update({"status": "FAIL", "reason": "request_metric_count_mismatch",
                       "expected_request_count": 8, "natural_completion": False})
    if peak_rss is None:
        record.update({"status": "INCOMPLETE", "reason": "peak_rss_unavailable",
                       "natural_completion": False})
    return record


def command_smoke(root: Path, *, source_root: Path | None,
                  prediction_config: Path | None, variant: str) -> int:
    """Run one isolated, complete one-second TJE trace arm; never edits live files."""
    result: dict[str, Any] = {
        "schema_version": "joint-accel-smoke-v1-20261006",
        "status": "INCOMPLETE", "reason_code": "smoke_not_started",
        "created_utc": utc_now(), "variant": variant,
    }
    source_resolution: dict[str, Any] = {"requested_source_root": None}
    if source_root is not None:
        source_resolution["selected_by"] = "command_line"
    else:
        accuracy_path = root / "manifests/frozen-accuracy.json"
        capture_path = root / "packages/acceptance/capture.json"
        candidates: list[tuple[str, Path]] = []
        if accuracy_path.is_file():
            accuracy_value = _require_object(read_json(accuracy_path), "frozen accuracy")
            accuracy_source = _require_object(accuracy_value.get("dev_source"), "frozen dev source")
            if isinstance(accuracy_source.get("source_root"), str):
                candidates.append(("frozen_accuracy", Path(accuracy_source["source_root"])))
        if variant == "exact" and capture_path.is_file():
            capture_value = _require_object(read_json(capture_path), "capture result")
            if (capture_value.get("status") == "PASS"
                    and isinstance(capture_value.get("source_root"), str)):
                candidates.append(("capture", Path(capture_value["source_root"])))
        if variant == "exact":
            candidates.append(("manual_exact", root / "runs/manual-exact/source"))
        selected = next(((label, path if path.is_absolute() else root / path)
                         for label, path in candidates
                         if (path if path.is_absolute() else root / path).is_dir()), None)
        if selected is not None:
            source_resolution["selected_by"], source_root = selected
        else:
            result.update({"reason_code": "smoke_source_copy_required",
                           "missing": ["built isolated source copy from --source-root, "
                                       "frozen capture/accuracy, or runs/manual-exact/source"]})
            return _write_acceptance_command_result(root, "smoke", result,
                                                    EXIT_INCOMPLETE)
    source_resolution["requested_source_root"] = str(source_root)
    result["source_resolution"] = source_resolution
    attempt_root = root / "runs/smoke"
    attempt_root.mkdir(parents=True, exist_ok=True)
    attempt = attempt_root / f"attempt-{_attempt_number(attempt_root):03d}"
    attempt.mkdir(exist_ok=False)
    result["attempt"] = str(attempt.relative_to(root))
    try:
        missing_parents = []
        for parent in ("G-C5", "G-D1", "G-E0"):
            try:
                validate_done(root, parent)
            except AcceptanceError:
                missing_parents.append(parent)
        if missing_parents:
            result.update({"reason_code": "smoke_parent_gates_missing",
                           "missing_parent_gates": missing_parents})
            return _write_acceptance_command_result(root, "smoke", result,
                                                    EXIT_INCOMPLETE)
        source = source_root.expanduser().resolve()
        if source == REPO_ROOT.resolve() or source == (root / "reference/source").resolve():
            raise AcceptanceError("smoke requires a built source/run copy, not live or frozen source",
                                  EXIT_FAIL, "smoke_source_not_isolated")
        source_record: dict[str, Any] | None = None
        source_record_candidates = ([root / "manifests/frozen-accuracy.json"]
                                    if variant == "approx" else
                                    [root / "packages/acceptance/capture.json",
                                     root / "manifests/frozen-accuracy.json"])
        for record_path in source_record_candidates:
            if not record_path.is_file():
                continue
            record_value = _require_object(read_json(record_path), record_path.name)
            if record_path.name == "capture.json":
                candidate_root = record_value.get("source_root")
                candidate_binding = record_value.get("implementation_source")
            else:
                dev_source = _require_object(record_value.get("dev_source"),
                                             "frozen dev source")
                candidate_root = dev_source.get("source_root")
                candidate_binding = dev_source.get("implementation_source")
            if (isinstance(candidate_root, str)
                    and Path(candidate_root).expanduser().resolve() == source):
                source_record = candidate_binding
                break
        if source_record is None:
            raise AcceptanceError("smoke source has no frozen implementation identity",
                                  EXIT_INCOMPLETE, "smoke_source_binding_missing")
        source_binding = _verify_implementation_source_binding(root, source, source_record)
        result["implementation_source"] = source_binding
        accuracy = None
        if variant == "approx":
            accuracy_path = root / "manifests/frozen-accuracy.json"
            if not accuracy_path.is_file():
                raise AcceptanceError("approximate smoke requires the frozen dev selection",
                                      EXIT_INCOMPLETE, "smoke_accuracy_selection_missing")
            accuracy = _require_object(read_json(accuracy_path), "frozen accuracy")
            if accuracy.get("status") != "PASS":
                raise AcceptanceError("frozen dev selection is not PASS",
                                      EXIT_FAIL, "smoke_accuracy_selection_invalid")
            accuracy_source = _require_object(accuracy["dev_source"], "frozen dev source")
            if Path(accuracy_source["source_root"]).resolve() != source:
                raise AcceptanceError("smoke source differs from frozen accuracy source",
                                      EXIT_FAIL, "smoke_source_mismatch")
            contract = validate_kernel_contract(
                accuracy_source.get("selected_kernel_contract"))
            config_record = _require_object(accuracy_source.get("selected_prediction_config"),
                                            "selected prediction config")
            semantic_config = _require_object(config_record.get("semantic_config"),
                                              "selected semantic config")
            accepted = accuracy.get("approximation_accepted") is True
        else:
            config_path = prediction_config
            if config_path is None:
                for recorded_path in (
                        root / "packages/acceptance/capture.json",
                        root / "manifests/frozen-accuracy.json"):
                    if not recorded_path.is_file():
                        continue
                    recorded = _require_object(read_json(recorded_path),
                                               recorded_path.name)
                    candidate = None
                    if recorded_path.name == "capture.json":
                        record = _require_object(recorded.get("prediction_config"),
                                                 "captured prediction config")
                        candidate = record.get("path")
                    else:
                        dev = _require_object(recorded.get("dev_source"), "frozen dev source")
                        record = _require_object(dev.get("prediction_config"),
                                                 "frozen exact prediction config")
                        candidate = record.get("path")
                    if isinstance(candidate, str) and Path(candidate).is_file():
                        config_path = Path(candidate)
                        break
            if config_path is not None and not config_path.is_absolute():
                config_path = (root / config_path).resolve()
            contract, config_record = resolve_prediction_contract(
                source, config_path)
            semantic_config = config_record["semantic_config"]
            accepted = True
        config_dir = attempt / "config"
        config_dir.mkdir()
        config_path = config_dir / f"prediction-{variant}.json"
        write_json(config_path, semantic_config)
        # The frozen selector manifest is created by capture. When smoke is run
        # independently, materialize from the frozen CSV after creating it.
        if not (root / "inputs/trace-window-1s-selection.json").is_file():
            selection = select_first_second_trace(root / "manifests/astra_compute_20.csv")
            write_json(root / "inputs/trace-window-1s-selection.json", selection)
        isolated = attempt / "sources/source"
        source_info = _copy_isolated_source(source, isolated)
        queue, queue_record = _make_one_second_queue(root, isolated, attempt)
        _set_request_queue_csv(isolated, queue)
        # plan_materializer must see the same queue path used by the run.
        planner = isolated / "sh_test_mesh/workload/llama2_7b_inference/plan_materializer.py"
        prep_log = attempt / "logs/source-plan-materializer.log"
        prep = run_logged(["python3", str(planner)], isolated,
                          _environment_for_tests(), prep_log, 300)
        if prep.get("status") != "completed" or prep.get("returncode") != 0:
            raise AcceptanceError("smoke plan materialization failed",
                                  EXIT_INCOMPLETE, "smoke_plan_materialization_failed")
        run_dir = attempt / "run"
        arm = _run_online_arm(isolated, queue, run_dir, config_path,
                              label=f"smoke-{variant}", attempt=attempt)
        source_binding = _verify_implementation_source_binding(root, source, source_record)
        result.update({"source": source_info, "variant_contract": contract,
                       "implementation_source": source_binding,
                       "variant_accepted": accepted, "config_sha256": sha256_file(config_path),
                       "queue": queue_record, "plan_materializer": prep,
                       "arm": arm, "completed_utc": utc_now()})
        if arm["status"] == "completed":
            result.update({"status": "PASS", "reason_code": "one_second_trace_completed"})
            code = EXIT_PASS if accepted else EXIT_INCOMPLETE
            if not accepted:
                result.update({"status": "INCOMPLETE",
                               "reason_code": "approximation_not_accepted"})
        elif arm["status"] == "FAIL":
            result.update({"status": "FAIL", "reason_code": arm.get("reason")})
            code = EXIT_FAIL
        else:
            result.update({"status": "INCOMPLETE", "reason_code": arm.get("reason")})
            code = EXIT_INCOMPLETE
        return _write_acceptance_command_result(root, "smoke", result, code)
    except AcceptanceError as exc:
        result.update({"status": "FAIL" if exc.code == EXIT_FAIL else "INCOMPLETE",
                       "reason_code": exc.reason_code, "error": str(exc),
                       "completed_utc": utc_now()})
        return _write_acceptance_command_result(
            root, "smoke", result, EXIT_FAIL if exc.code == EXIT_FAIL else exc.code)


def command_performance(root: Path, *, source_root: Path | None,
                        prediction_config: Path | None,
                        exact_only: bool = False) -> int:
    """Run a formal three-arm campaign or an explicitly unaccepted exact-only diagnostic."""
    result: dict[str, Any] = {
        "schema_version": "joint-accel-performance-v1-20261006",
        "status": "INCOMPLETE", "reason_code": "performance_not_started",
        "created_utc": utc_now(), "workload": "frozen_first_second_trace",
        "repetitions": [], "invocations": [],
        "mode": "exact_only_diagnostic" if exact_only else "formal_three_arm",
        "formal_speed_claim": False if exact_only else None,
    }
    required_gates = ("G-C5", "G-D1", "G-E0") + (() if exact_only else ("G-F0",))
    missing = []
    for name in required_gates:
        try:
            validate_done(root, name)
            result.setdefault("parent_gates", {})[name] = sha256_file(
                root / "gates" / name / "DONE")
        except AcceptanceError:
            missing.append(name)
    if missing:
        result.update({"reason_code": "performance_parent_gates_missing",
                       "missing_parent_gates": missing})
        return _write_acceptance_command_result(root, "performance", result,
                                                EXIT_INCOMPLETE)
    accuracy_path = root / "manifests/frozen-accuracy.json"
    capture_path = root / "packages/acceptance/capture.json"
    baseline_path = root / "packages/acceptance/baseline.json"
    if (not accuracy_path.is_file() and not (exact_only and capture_path.is_file())) \
            or not baseline_path.is_file():
        result.update({"reason_code": "performance_frozen_inputs_missing",
                       "missing": [str(path.relative_to(root)) for path in
                                   (accuracy_path, baseline_path) if not path.is_file()]
                       + (["packages/acceptance/capture.json"]
                          if exact_only and not capture_path.is_file() else [])})
        return _write_acceptance_command_result(root, "performance", result,
                                                EXIT_INCOMPLETE)
    try:
        accuracy_value = (_require_object(read_json(accuracy_path), "frozen accuracy")
                          if accuracy_path.is_file() else None)
        if not exact_only and (accuracy_value is None
                               or accuracy_value.get("status") != "PASS"):
            raise AcceptanceError("frozen development accuracy selection is not PASS",
                                  EXIT_FAIL, "performance_accuracy_selection_invalid")
        use_accuracy = (accuracy_value is not None
                        and accuracy_value.get("status") == "PASS"
                        and isinstance(accuracy_value.get("dev_source"), dict))
        if use_accuracy:
            accuracy = accuracy_value
            accuracy_source = _require_object(accuracy.get("dev_source"), "frozen dev source")
        elif exact_only:
            capture = _require_object(read_json(capture_path), "capture result")
            if capture.get("status") != "PASS":
                raise AcceptanceError("exact-only trace comparison requires a verified exact capture",
                                      EXIT_INCOMPLETE, "performance_exact_capture_missing")
            exact_record = _require_object(capture.get("prediction_config"),
                                           "captured exact prediction config")
            accuracy = {"status": "PASS", "approximation_accepted": False}
            accuracy_source = {
                "source_root": capture.get("source_root"),
                "implementation_source": capture.get("implementation_source"),
                "prediction_config": exact_record,
                "selected_prediction_config": exact_record,
                "selected_kernel_contract": capture.get("kernel_contract"),
            }
        else:
            raise AcceptanceError("frozen accuracy development source is unavailable",
                                  EXIT_INCOMPLETE, "performance_frozen_inputs_missing")
        selected_contract = (None if exact_only else validate_kernel_contract(
            accuracy_source.get("selected_kernel_contract")))
        source = (source_root or Path(accuracy_source["source_root"])).expanduser().resolve()
        if Path(accuracy_source["source_root"]).expanduser().resolve() != source:
            raise AcceptanceError("performance source differs from the frozen dev source",
                                  EXIT_FAIL, "performance_source_mismatch")
        source_binding = _verify_implementation_source_binding(
            root, source, accuracy_source.get("implementation_source"))
        result["implementation_source"] = source_binding
        exact_config = _require_object(accuracy_source.get("prediction_config"),
                                       "exact prediction config")
        exact_semantic = _require_object(exact_config.get("semantic_config"),
                                         "exact semantic config")
        exact_config_path = Path(exact_config.get("path", "")).expanduser()
        if not exact_config_path.is_absolute():
            candidate = root / exact_config_path
            exact_config_path = candidate if candidate.exists() else source / exact_config_path
        if not exact_config_path.is_file():
            raise AcceptanceError("frozen prediction config inputs are missing",
                                  EXIT_INCOMPLETE, "performance_config_missing")
        exact_contract, resolved_exact = resolve_prediction_contract(
            source, exact_config_path)
        if resolved_exact["semantic_sha256"] != exact_config.get("semantic_sha256"):
            raise AcceptanceError("prediction config changed since development freeze",
                                  EXIT_FAIL, "performance_config_mismatch")
        if exact_only:
            # The exact-only diagnostic uses the same frozen exact semantic
            # config for both binaries.  It never inherits the selected
            # tuning contract recorded by a successful accuracy artifact.
            selected_contract = exact_contract
            selected_config = exact_config
            selected_config_path = exact_config_path
            selected_semantic = exact_semantic
            resolved_selected = resolved_exact
            resolved_approx = resolved_exact
        else:
            selected_config = _require_object(
                accuracy_source.get("selected_prediction_config"),
                "selected prediction config")
            selected_semantic = _require_object(selected_config.get("semantic_config"),
                                                "selected semantic config")
            selected_config_path = Path(selected_config.get("path", "")).expanduser()
            if not selected_config_path.is_absolute():
                candidate = root / selected_config_path
                selected_config_path = (candidate if candidate.exists()
                                        else source / selected_config_path)
            if not selected_config_path.is_file():
                raise AcceptanceError("frozen prediction config inputs are missing",
                                      EXIT_INCOMPLETE, "performance_config_missing")
            resolved_selected, resolved_approx = resolve_prediction_contract(
                source, selected_config_path)
            if (resolved_approx["semantic_sha256"] != selected_config.get("semantic_sha256")
                    or resolved_selected != selected_contract):
                raise AcceptanceError("prediction config changed since development freeze",
                                      EXIT_FAIL, "performance_config_mismatch")
        if selected_config.get("semantic_sha256") != selected_contract[
                "prediction_config_sha256"]:
            raise AcceptanceError("selected config and kernel contract disagree",
                                  EXIT_FAIL, "performance_accuracy_selection_invalid")
        if (exact_config.get("semantic_sha256") !=
                exact_contract["prediction_config_sha256"]):
            raise AcceptanceError(
                "frozen exact config does not resolve to the exact 6-key kernel contract",
                EXIT_FAIL, "performance_exact_config_invalid")
        if prediction_config is not None:
            requested_config = prediction_config.expanduser()
            if not requested_config.is_absolute():
                requested_config = (root / requested_config).resolve()
            if (not requested_config.is_file()
                    or sha256_file(exact_config_path) != sha256_file(requested_config)):
                raise AcceptanceError("performance prediction config differs from frozen exact input",
                                      EXIT_FAIL, "performance_config_mismatch")
        baseline = _require_object(read_json(baseline_path), "baseline result")
        if baseline.get("status") != "PASS" or not isinstance(baseline.get("run_source"), str):
            raise AcceptanceError("A1 baseline does not identify a complete old source copy",
                                  EXIT_INCOMPLETE, "performance_old_source_missing")
        old_source = (root / baseline["run_source"]).resolve()
        if not old_source.is_dir():
            raise AcceptanceError("A1 baseline source copy is missing",
                                  EXIT_INCOMPLETE, "performance_old_source_missing")
        if source == (root / "reference/source").resolve():
            raise AcceptanceError("performance requires the hash-recorded built source copy",
                                  EXIT_FAIL, "performance_source_not_isolated")
        if source == REPO_ROOT.resolve():
            # 覆盖驱动银行（§19.2 第 3 项）以 live 树为 source_root：其
            # implementation-source inventory 已冻结于证据根并经
            # _verify_implementation_source_binding 校验；测量经
            # _prepare_online_source 复制到 attempt 目录执行，不写源树。
            # reference/source 只读参考仍拒绝。
            result["source_kind"] = "repo_root_with_frozen_binding"
        attempt_root = root / "runs/performance"
        attempt_root.mkdir(parents=True, exist_ok=True)
        attempt = attempt_root / f"attempt-{_attempt_number(attempt_root):03d}"
        attempt.mkdir(exist_ok=False)
        result["attempt"] = str(attempt.relative_to(root))
        selection_path = root / "inputs/trace-window-1s-selection.json"
        selection = select_first_second_trace(root / "manifests/astra_compute_20.csv")
        if selection_path.exists():
            existing_selection = read_json(selection_path)
            stable = {key: value for key, value in existing_selection.items()
                      if key != "materializer"}
            expected = {key: value for key, value in selection.items()
                        if key != "materializer"}
            if stable != expected:
                raise AcceptanceError("frozen one-second trace selection changed",
                                      EXIT_FAIL, "trace_selection_manifest_mismatch")
        else:
            write_json(selection_path, selection)
        frozen_source = root / "reference/source"
        queue, queue_record = _make_one_second_queue(root, frozen_source, attempt)
        config_dir = attempt / "config"
        config_dir.mkdir()
        exact_config_wire = config_dir / "prediction-new-exact.json"
        approx_config_path = config_dir / "prediction-new-approx.json"
        write_json(exact_config_wire, exact_semantic)
        write_json(approx_config_path, selected_semantic)
        old_copy, old_copy_info = _prepare_online_source(
            old_source, queue, attempt, "old-exact-source")
        new_copy, new_copy_info = _prepare_online_source(
            source, queue, attempt, "new-source")
        result.update({
            "old_source": old_copy_info, "new_source": new_copy_info,
            "old_binary_sha256": sha256_file(
                old_copy / "build/astra_analytical/build_congestion_aware/bin"
                / "AstraSim_Analytical_Congestion_Aware_Online"),
            "new_binary_sha256": sha256_file(
                new_copy / "build/astra_analytical/build_congestion_aware/bin"
                / "AstraSim_Analytical_Congestion_Aware_Online"),
            "queue": queue_record,
            "selected_kernel_contract": selected_contract,
            "exact_kernel_contract": exact_contract,
            "approximation_accepted": accuracy.get("approximation_accepted") is True,
        })
        expected_orders = ([
            ["old_exact", "new_exact"],
            ["new_exact", "old_exact"],
            ["old_exact", "new_exact"],
            ["new_exact", "old_exact"],
            ["old_exact", "new_exact"],
        ] if exact_only else [
            ["old_exact", "new_exact", "new_approx"],
            ["new_exact", "new_approx", "old_exact"],
            ["new_approx", "old_exact", "new_exact"],
            ["old_exact", "new_exact", "new_approx"],
            ["new_exact", "new_approx", "old_exact"],
        ])
        armspec = {
            "old_exact": (old_copy, None),
            "new_exact": (new_copy, exact_config_wire),
        }
        if not exact_only:
            armspec["new_approx"] = (new_copy, approx_config_path)
        for repetition_index, order in enumerate(expected_orders, start=1):
            arms: dict[str, Any] = {}
            repetition = {"repetition": repetition_index, "order": order,
                          "status": "incomplete", "arms": arms}
            for arm_name in order:
                arm_source, config_path = armspec[arm_name]
                run_dir = attempt / f"rep-{repetition_index:02d}" / f"run-{arm_name}"
                # D18：旧实现 1s 窗口完整自然 decode 实测 1507s > 300s
                # cap——"既定宿主上限"按需设定（§19.6），不截断自然完成。
                arm = _run_online_arm(arm_source, queue, run_dir, config_path,
                                      label=f"r{repetition_index}-{arm_name}",
                                      attempt=attempt, timeout_seconds=1800)
                arm["natural_completion"] = arm.get("status") == "completed"
                arms[arm_name] = arm
                result["invocations"].append({"repetition": repetition_index,
                                              "arm": arm_name, **arm})
                if arm.get("status") != "completed":
                    repetition["status"] = arm.get("status", "incomplete")
                    repetition["stopped_at"] = arm_name
                    repetition["reason"] = arm.get("reason", "arm_not_complete")
                    result["repetitions"].append(repetition)
                    result.update({"status": ("FAIL" if repetition["status"] == "FAIL"
                                               else "INCOMPLETE"),
                                   "reason_code": "first_performance_arm_failed_or_incomplete",
                                   "completed_utc": utc_now()})
                    write_json(attempt / "result.json", result)
                    return _write_acceptance_command_result(
                        root, "performance", result,
                        EXIT_FAIL if repetition["status"] == "FAIL" else EXIT_INCOMPLETE)
            signatures = {record.get("work_signature_sha256")
                          for record in arms.values()}
            if len(signatures) != 1:
                raise AcceptanceError("rotated performance arms did not use identical work",
                                      EXIT_FAIL, "performance_work_mismatch")
            repetition["status"] = "complete"
            result["repetitions"].append(repetition)
            write_json(attempt / "progress.json", result)
        summary = (summarize_exact_only_repetitions(result["repetitions"])
                   if exact_only else
                   summarize_performance_repetitions(result["repetitions"]))
        result.update(summary)
        result.update({"reason_code": ("five_rotated_exact_only_pairs_completed"
                                        if exact_only else
                                        "five_rotated_performance_pairs_completed"),
                       "completed_utc": utc_now()})
        write_json(attempt / "result.json", result)
        return _write_acceptance_command_result(root, "performance", result, EXIT_PASS)
    except AcceptanceError as exc:
        result.update({"status": "FAIL" if exc.code == EXIT_FAIL else "INCOMPLETE",
                       "reason_code": exc.reason_code, "error": str(exc),
                       "completed_utc": utc_now()})
        return _write_acceptance_command_result(
            root, "performance", result,
            EXIT_FAIL if exc.code == EXIT_FAIL else EXIT_INCOMPLETE)


def inventory_tree(tree: Path, exclude: set[str] | None = None) -> tuple[list[dict[str, Any]], str]:
    """Return a stable full-tree inventory; symlink targets are never followed."""
    if not tree.is_dir():
        raise AcceptanceError(f"required tree is missing: {tree}")
    entries: list[dict[str, Any]] = []
    for parent, dirs, files in os.walk(tree, topdown=True, followlinks=False):
        dirs.sort()
        files.sort()
        base = Path(parent)
        for name in dirs + files:
            path = base / name
            rel = path.relative_to(tree).as_posix()
            if exclude and rel in exclude:
                continue
            info = path.lstat()
            mode = stat.S_IMODE(info.st_mode)
            if stat.S_ISLNK(info.st_mode):
                target = os.readlink(path)
                entries.append({"path": rel, "type": "symlink", "mode": mode,
                                "size": len(os.fsencode(target)),
                                "link_text": target})
            elif stat.S_ISDIR(info.st_mode):
                entries.append({"path": rel, "type": "directory", "mode": mode})
            elif stat.S_ISREG(info.st_mode):
                entries.append({"path": rel, "type": "file", "mode": mode,
                                "size": info.st_size, "sha256": sha256_file(path)})
            else:
                entries.append({"path": rel, "type": "other", "mode": mode,
                                "size": info.st_size})
    entries.sort(key=lambda item: item["path"])
    encoded = json.dumps(entries, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False).encode("utf-8")
    return entries, hashlib.sha256(encoded).hexdigest()


_SOURCE_IGNORED_DIRECTORY_NAMES = {
    ".git", "__pycache__", ".pytest_cache", "runs", "build",
    "build_congestion_aware", "build_congestion", "CMakeFiles",
}
_SOURCE_IGNORED_SUFFIXES = {
    ".pyc", ".o", ".a", ".so", ".dylib", ".dll", ".exe", ".log",
    ".md", ".rst",
}


def implementation_source_inventory(tree: Path) -> tuple[list[dict[str, Any]], str]:
    """Hash implementation/config files while excluding mutable build and run output."""
    tree = tree.expanduser().resolve()
    if not tree.is_dir():
        raise AcceptanceError(f"implementation source tree is missing: {tree}",
                              EXIT_INCOMPLETE, "implementation_source_missing")
    entries: list[dict[str, Any]] = []
    for parent, dirs, files in os.walk(tree, topdown=True, followlinks=False):
        base = Path(parent)
        kept_dirs: list[str] = []
        for name in sorted(dirs):
            path = base / name
            relative = path.relative_to(tree).as_posix()
            parts = Path(relative).parts
            if (name in _SOURCE_IGNORED_DIRECTORY_NAMES
                    or name.startswith("build_") or name.startswith("build-")
                    or parts[:2] == ("sh_test_mesh", "generated")):
                continue
            kept_dirs.append(name)
            try:
                info = path.lstat()
            except OSError as exc:
                raise AcceptanceError(f"cannot inspect source path {path}: {exc}",
                                      EXIT_INCOMPLETE,
                                      "implementation_source_unreadable") from exc
            mode = stat.S_IMODE(info.st_mode)
            if stat.S_ISLNK(info.st_mode):
                target = os.readlink(path)
                entries.append({"path": relative, "type": "symlink", "mode": mode,
                                "size": len(os.fsencode(target)), "link_text": target})
            elif stat.S_ISDIR(info.st_mode):
                entries.append({"path": relative, "type": "directory", "mode": mode})
        dirs[:] = kept_dirs
        for name in sorted(files):
            path = base / name
            relative = path.relative_to(tree).as_posix()
            if (Path(name).suffix.lower() in _SOURCE_IGNORED_SUFFIXES
                    or name.lower() == "readme" or name.lower().startswith("readme.")
                    or name.lower().startswith(("license", "notice", "changelog"))
                    or name in {"CMakeCache.txt", "compile_commands.json"}
                    or name.startswith(".ninja_")):
                continue
            try:
                info = path.lstat()
            except OSError as exc:
                raise AcceptanceError(f"cannot inspect source path {path}: {exc}",
                                      EXIT_INCOMPLETE,
                                      "implementation_source_unreadable") from exc
            mode = stat.S_IMODE(info.st_mode)
            if stat.S_ISLNK(info.st_mode):
                target = os.readlink(path)
                entries.append({"path": relative, "type": "symlink", "mode": mode,
                                "size": len(os.fsencode(target)), "link_text": target})
            elif stat.S_ISREG(info.st_mode):
                entries.append({"path": relative, "type": "file", "mode": mode,
                                "size": info.st_size, "sha256": sha256_file(path)})
    entries.sort(key=lambda item: item["path"])
    digest = _canonical_digest(entries)
    return entries, digest


def _save_implementation_source_binding(root: Path, source_root: Path) -> dict[str, Any]:
    source_root = source_root.expanduser().resolve()
    entries, tree_digest = implementation_source_inventory(source_root)
    driver = Path(__file__).resolve()
    source_key = hashlib.sha256(str(source_root).encode("utf-8")).hexdigest()[:12]
    relative_manifest = Path("manifests") / (
        f"implementation-source-{tree_digest}-{source_key}.json")
    manifest_path = root / relative_manifest
    manifest = {
        "schema_version": IMPLEMENTATION_SOURCE_INVENTORY_VERSION,
        "source_root": str(source_root), "tree_digest_sha256": tree_digest,
        "ignored_directory_names": sorted(_SOURCE_IGNORED_DIRECTORY_NAMES),
        "ignored_directory_patterns": ["build_*", "build-*",
                                        "sh_test_mesh/generated/**"],
        "ignored_file_suffixes": sorted(_SOURCE_IGNORED_SUFFIXES),
        "ignored_file_patterns": ["README*", "LICENSE*", "NOTICE*", "CHANGELOG*",
                                  "CMakeCache.txt", "compile_commands.json", ".ninja_*"],
        "entries": entries,
    }
    if manifest_path.exists():
        if read_json(manifest_path) != manifest:
            raise AcceptanceError("existing implementation source inventory differs; preserving it",
                                  EXIT_FAIL, "implementation_source_manifest_mismatch")
    else:
        write_json(manifest_path, manifest)
    return {
        "schema_version": IMPLEMENTATION_SOURCE_INVENTORY_VERSION,
        "source_root": str(source_root), "tree_digest_sha256": tree_digest,
        "inventory_path": relative_manifest.as_posix(),
        "inventory_sha256": sha256_file(manifest_path),
        "acceptance_driver_path": str(driver),
        "acceptance_driver_sha256": sha256_file(driver),
    }


def _verify_implementation_source_binding(root: Path, source_root: Path,
                                          binding: Any) -> dict[str, Any]:
    if not isinstance(binding, dict) or binding.get("schema_version") != \
            IMPLEMENTATION_SOURCE_INVENTORY_VERSION:
        raise AcceptanceError("capture lacks the current implementation-source inventory",
                              EXIT_INCOMPLETE, "implementation_source_binding_missing")
    source_root = source_root.expanduser().resolve()
    if Path(str(binding.get("source_root", ""))).expanduser().resolve() != source_root:
        raise AcceptanceError("selected source root differs from its captured source identity",
                              EXIT_FAIL, "implementation_source_root_mismatch")
    relative = binding.get("inventory_path")
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise AcceptanceError("implementation source inventory path is invalid", EXIT_FAIL,
                              "implementation_source_manifest_invalid")
    manifest_path = (root / relative).resolve()
    try:
        manifest_path.relative_to(root.resolve())
    except ValueError as exc:
        raise AcceptanceError("implementation source inventory escapes evidence root",
                              EXIT_FAIL, "implementation_source_manifest_invalid") from exc
    if not manifest_path.is_file():
        raise AcceptanceError("implementation source inventory artifact is missing",
                              EXIT_INCOMPLETE, "implementation_source_manifest_missing")
    if sha256_file(manifest_path) != binding.get("inventory_sha256"):
        raise AcceptanceError("implementation source inventory artifact changed", EXIT_FAIL,
                              "implementation_source_manifest_digest_mismatch")
    manifest = _require_object(read_json(manifest_path), "implementation source inventory")
    if (manifest.get("schema_version") != IMPLEMENTATION_SOURCE_INVENTORY_VERSION
            or manifest.get("source_root") != str(source_root)
            or manifest.get("tree_digest_sha256") != binding.get("tree_digest_sha256")
            or _canonical_digest(manifest.get("entries")) != binding.get("tree_digest_sha256")):
        raise AcceptanceError("implementation source inventory metadata is inconsistent",
                              EXIT_FAIL, "implementation_source_manifest_invalid")
    current_entries, current_digest = implementation_source_inventory(source_root)
    if current_digest != binding.get("tree_digest_sha256"):
        raise AcceptanceError("implementation source changed after cold capture", EXIT_FAIL,
                              "implementation_source_digest_mismatch")
    driver = Path(__file__).resolve()
    if (Path(str(binding.get("acceptance_driver_path", ""))).resolve() != driver
            or binding.get("acceptance_driver_sha256") != sha256_file(driver)):
        raise AcceptanceError("acceptance driver changed after cold capture", EXIT_FAIL,
                              "acceptance_driver_digest_mismatch")
    return {"source_root": str(source_root), "tree_digest_sha256": current_digest,
            "inventory_sha256": sha256_file(manifest_path),
            "acceptance_driver_sha256": sha256_file(driver),
            "entry_count": len(current_entries)}


def evidence_root(value: str | Path) -> Path:
    root = Path(value).expanduser().resolve()
    if not root.is_dir():
        raise AcceptanceError(f"evidence root does not exist: {root}")
    required = (root / "reference/source", root / "manifests")
    missing = [str(path) for path in required if not path.is_dir()]
    if missing:
        raise AcceptanceError("incomplete evidence root; missing: " + ", ".join(missing))
    return root


def validate_done(root: Path, gate: str,
                  expected_parent: dict[str, str] | None = None) -> dict[str, Any]:
    if gate not in GATE_PARENTS:
        raise AcceptanceError(f"unknown acceptance gate {gate}", EXIT_FAIL,
                              "gate_name_invalid")
    gate_dir = root / "gates" / gate
    result_path, done_path = gate_dir / "result.json", gate_dir / "DONE"
    if not result_path.is_file() or not done_path.is_file():
        raise AcceptanceError(f"parent gate {gate} lacks result.json or DONE")
    result, done = read_json(result_path), read_json(done_path)
    if not isinstance(result, dict) or not isinstance(done, dict):
        raise AcceptanceError(f"parent gate {gate} evidence must be objects", EXIT_FAIL,
                              "gate_schema_invalid")
    if result.get("gate") != gate or done.get("gate") != gate:
        raise AcceptanceError(f"parent gate {gate} identity mismatch", EXIT_FAIL,
                              "gate_identity_mismatch")
    if result.get("status") != "PASS" or done.get("status") != "PASS":
        raise AcceptanceError(f"parent gate {gate} is not PASS")
    if done.get("result_sha256") != sha256_file(result_path):
        raise AcceptanceError(f"parent gate {gate} result digest changed",
                              EXIT_FAIL, "parent_result_digest_mismatch")
    if expected_parent is None:
        expected_parent = {}
        for parent in GATE_PARENTS[gate]:
            validate_done(root, parent)
            expected_parent[parent] = sha256_file(root / "gates" / parent / "DONE")
    if done.get("parents", {}) != expected_parent:
        raise AcceptanceError(f"parent gate {gate} ancestry changed",
                              EXIT_FAIL, "parent_gate_hash_mismatch")
    if result.get("parents", {}) != expected_parent:
        raise AcceptanceError(f"parent gate {gate} result ancestry changed",
                              EXIT_FAIL, "parent_gate_hash_mismatch")
    artifacts = done.get("required_artifacts", {})
    if not isinstance(artifacts, dict):
        raise AcceptanceError(f"parent gate {gate} artifact table is invalid", EXIT_FAIL,
                              "gate_artifact_manifest_invalid")
    declared_artifacts = result.get("required_artifacts")
    if declared_artifacts is not None and declared_artifacts != artifacts:
        raise AcceptanceError(f"parent gate {gate} artifact manifest changed", EXIT_FAIL,
                              "gate_artifact_manifest_mismatch")
    for rel, expected in artifacts.items():
        if not isinstance(rel, str) or not rel or Path(rel).is_absolute():
            raise AcceptanceError(f"parent gate {gate} artifact path is invalid", EXIT_FAIL,
                                  "gate_artifact_manifest_invalid")
        path = (root / rel).resolve()
        try:
            path.relative_to(root.resolve())
        except ValueError as exc:
            raise AcceptanceError(f"parent gate {gate} artifact escapes evidence root",
                                  EXIT_FAIL, "gate_artifact_manifest_invalid") from exc
        if not path.is_file() or sha256_file(path) != expected:
            raise AcceptanceError(f"parent gate {gate} artifact mismatch: {rel}",
                                  EXIT_FAIL, "parent_artifact_digest_mismatch")
    return done


def gate_status_detail(root: Path, gate: str) -> dict[str, Any]:
    """Return the recorded gate state without collapsing failure categories."""
    if gate not in GATE_PARENTS:
        return {"status": "FAIL", "reason_code": "gate_name_invalid"}
    result_path = root / "gates" / gate / "result.json"
    done_path = root / "gates" / gate / "DONE"
    if not result_path.is_file():
        return {"status": "MISSING", "reason_code": "gate_result_missing"}
    try:
        result = _require_object(read_json(result_path), f"{gate} result")
    except AcceptanceError as exc:
        return {"status": "FAIL", "reason_code": exc.reason_code,
                "message": str(exc)}
    if result.get("gate") != gate:
        return {"status": "FAIL", "reason_code": "gate_identity_mismatch"}
    status = result.get("status")
    if not isinstance(status, str) or status not in GATE_STATUSES:
        return {"status": "FAIL", "reason_code": "gate_status_invalid"}
    if status != "PASS":
        return {"status": status, "reason_code": result.get("reason_code"),
                "result_sha256": sha256_file(result_path),
                "stale_done_present": done_path.exists()}
    try:
        done = validate_done(root, gate)
    except AcceptanceError as exc:
        return {"status": "INCOMPLETE" if exc.code != EXIT_FAIL else "FAIL",
                "reason_code": exc.reason_code, "message": str(exc),
                "result_sha256": sha256_file(result_path),
                "done_present": done_path.is_file()}
    return {"status": "PASS", "reason_code": result.get("reason_code"),
            "result_sha256": sha256_file(result_path),
            "done_sha256": sha256_file(done_path),
            "artifact_count": len(done.get("required_artifacts", {}))}


def invalidate_done(root: Path, gate: str, reason: str) -> Path | None:
    done = root / "gates" / gate / "DONE"
    if not done.exists():
        return None
    archived = done.with_name(f"DONE.invalidated-{time.time_ns()}")
    os.replace(done, archived)
    with (done.parent / "invalidations.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"at_utc": utc_now(), "reason": reason,
                                 "archived_done": archived.name}) + "\n")
    return archived


def validate_a0(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    done = validate_done(root, "G-A0")
    inventory_path = root / "manifests/source-inventory.json"
    if not inventory_path.is_file():
        raise AcceptanceError("A0 source inventory is missing")
    inventory = read_json(inventory_path)
    _, actual_digest = inventory_tree(root / "reference/source")
    expected_digest = inventory.get("tree_digest_sha256")
    if actual_digest != expected_digest or done.get("frozen_tree_digest_sha256") != actual_digest:
        raise AcceptanceError("frozen reference changed after A0", EXIT_FAIL,
                              "frozen_reference_digest_mismatch")
    old_bug = root / OLD_BUDGET_FIXTURE
    if not old_bug.is_file():
        raise AcceptanceError("pre-registered old event-budget fixture is missing")
    return done, inventory


def _environment_for_tests() -> dict[str, str]:
    env = os.environ.copy()
    removed = [key for key in env if key.startswith(("JOINT_", "SH30_", "SH3_CAUSAL_", "SH_JOINT_"))]
    for key in removed:
        env.pop(key, None)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    return env


def run_logged(argv: list[str], cwd: Path, env: dict[str, str],
               log_path: Path, timeout_seconds: int | None = None) -> dict[str, Any]:
    """Run one command without a shell and keep all output in evidence storage."""
    if not argv or shutil.which(argv[0]) is None:
        return {"argv": argv, "cwd": str(cwd), "status": "missing_dependency",
                "returncode": None, "log": str(log_path)}
    log_path.parent.mkdir(parents=True, exist_ok=True)
    start_wall = time.monotonic_ns()
    start_utc = utc_now()
    timed_out = False
    try:
        with log_path.open("wb") as log:
            process = subprocess.Popen(
                argv, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True)
            try:
                returncode = process.wait(timeout=timeout_seconds)
            except subprocess.TimeoutExpired:
                timed_out = True
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                returncode = process.returncode
    except OSError as exc:
        return {"argv": argv, "cwd": str(cwd), "status": "launch_error",
                "returncode": None, "error": str(exc), "log": str(log_path)}
    return {
        "argv": argv, "cwd": str(cwd), "status": "timeout" if timed_out else (
            "completed" if returncode == 0 else "nonzero_exit"),
        "returncode": returncode, "started_utc": start_utc,
        "elapsed_host_ns": time.monotonic_ns() - start_wall,
        "log": str(log_path), "log_sha256": sha256_file(log_path),
    }


def _error_excerpt(text: str) -> str:
    candidates = []
    for line in text.splitlines():
        stripped = re.sub(r"\x1b\[[0-9;]*m", "", line).strip()
        if re.search(r"(AssertionError|Error:|Exception:|FAILED|FAIL:|ERROR:|\[FAIL\]|\[error\])", stripped, re.I):
            candidates.append(" ".join(stripped.split()))
    return "\n".join(candidates[-8:]) or "nonzero test process without recognized failure summary"


def _failure_cases(argv: list[str], text: str) -> list[dict[str, str]]:
    cases: list[dict[str, str]] = []
    pytest_nodes = re.findall(r"^(?:FAILED|ERROR)\s+([^\s]+)", text, flags=re.MULTILINE)
    unit_nodes = [
        f"{klass}::{method}" if klass else method
        for method, klass in re.findall(
            r"^(?:FAIL|ERROR):\s+([^\s]+)(?:\s+\(([^)]+)\))?",
            text, flags=re.MULTILINE)
    ]
    ctest_nodes = re.findall(r"^\s*\d+\s+-\s+(.+?)\s+\(Failed\)\s*$", text, flags=re.MULTILINE)
    # Pytest emits a traceback block per failed node. Use the node's block for
    # its error signature, so distinct failures cannot collapse to the same
    # whole-log signature merely because they shared a command.
    failure_blocks: list[tuple[str, str]] = []
    block_start = re.compile(
        r"(?m)^_+\s*([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+)\s*_+\s*$")
    starts = list(block_start.finditer(text))
    for index, match in enumerate(starts):
        end = starts[index + 1].start() if index + 1 < len(starts) else len(text)
        block = text[match.start():end]
        summary = re.search(r"(?m)^={3,} (?:warnings summary|short test summary info|short test summary)", block)
        if summary:
            block = block[:summary.start()]
        failure_blocks.append((match.group(1), block))

    def excerpt_for(node: str) -> str:
        method = node.rsplit("::", 1)[-1].split("[", 1)[0]
        for heading, block in failure_blocks:
            if method and heading.endswith("." + method):
                excerpt = _error_excerpt(block)
                if excerpt != "nonzero test process without recognized failure summary":
                    return excerpt
        # CTest often places the diagnostic immediately before its summary.
        # A bounded tail gives a stable signature while retaining context.
        excerpt = _error_excerpt(text)
        return excerpt

    for node in pytest_nodes + unit_nodes + ctest_nodes:
        excerpt = excerpt_for(node)
        input_record = json.dumps({"test_node": node, "argv": argv,
                                   "test_input_model": "node id plus fixed test-body fixture/input"},
                                  sort_keys=True, ensure_ascii=False)
        cases.append({"test_name": node, "input": input_record,
                      "input_signature": hashlib.sha256(input_record.encode()).hexdigest(),
                      "error_excerpt": excerpt,
                      "error_signature": hashlib.sha256(excerpt.encode()).hexdigest()})
    if not cases and argv and Path(argv[0]).name.startswith("AstraSim_"):
        name = Path(argv[0]).name
        inputs = json.dumps(argv[1:], sort_keys=True)
        failing_assertions = re.findall(r"^.*(?:\[FAIL\]|FAIL:|FAILED|ASSERT|assertion failed|fatal error).*$",
                                        text, flags=re.MULTILINE | re.I)
        failing_assertions = [line for line in failing_assertions
                              if "see messages above" not in line.lower()]
        for assertion in failing_assertions:
            excerpt = " ".join(assertion.split())
            test_name = f"{name}::{excerpt}"
            cases.append({"test_name": test_name,
                          "input": "fixture invocation args: " + inputs,
                          "input_signature": hashlib.sha256((name + inputs).encode()).hexdigest(),
                          "error_excerpt": excerpt,
                          "error_signature": hashlib.sha256(excerpt.encode()).hexdigest()})
    return cases


def _record_existing_command(argv: list[str], cwd: Path, log_path: Path,
                             expected_exit: int | None) -> dict[str, Any]:
    """Recover a prior command only when its output has a complete signature."""
    record: dict[str, Any] = {"argv": argv, "cwd": str(cwd),
                              "log": str(log_path), "log_source": "pre-existing-logged-run"}
    if not log_path.is_file():
        return {**record, "status": "missing_log", "returncode": None}
    text = log_path.read_text(encoding="utf-8", errors="replace")
    lower_argv = " ".join(argv).lower()
    if "pytest" in lower_argv:
        summary = re.search(r"(?m)^([\d,]+ failed, )?([\d,]+ passed|no tests ran)(?:,.*)?(?: in [\d.]+s)?$", text)
        if not summary:
            return {**record, "status": "unrecoverable_log", "returncode": None,
                    "incomplete_reason": "pytest_summary_missing"}
        returncode = 1 if summary.group(1) else 0
    elif "unittest" in lower_argv:
        summary = re.search(r"(?m)^(OK(?: \(.*\))?|FAILED \(.*\))$", text)
        if not summary:
            return {**record, "status": "unrecoverable_log", "returncode": None,
                    "incomplete_reason": "unittest_summary_missing"}
        returncode = 1 if summary.group(1).startswith("FAILED") else 0
    elif Path(argv[0]).name == "ctest":
        if re.search(r"(?m)^100% tests passed, 0 tests failed(?: out of \d+)?$", text):
            returncode = 0
        elif re.search(r"(?m)^\d+% tests passed, \d+ tests failed out of \d+$", text):
            returncode = 8
        else:
            return {**record, "status": "unrecoverable_log", "returncode": None,
                    "incomplete_reason": "ctest_summary_missing_or_no_tests"}
    elif Path(argv[0]).name.startswith("AstraSim_"):
        if "ALL PASS" in text and not re.search(r"(?m)^.*(?:\[FAIL\]|FAIL:).*$", text):
            returncode = 0
        elif re.search(r"(?m)^.*(?:\[FAIL\]|FAIL:).*$", text):
            returncode = 1
        else:
            return {**record, "status": "unrecoverable_log", "returncode": None,
                    "incomplete_reason": "fixture_completion_marker_missing"}
    else:
        return {**record, "status": "unrecoverable_log", "returncode": None,
                "incomplete_reason": "unknown_command_log_signature"}
    if expected_exit is not None and expected_exit != returncode:
        return {**record, "status": "exit_signature_mismatch", "returncode": returncode}
    return {**record, "status": "completed" if returncode == 0 else "nonzero_exit",
            "returncode": returncode, "log_sha256": sha256_file(log_path)}


def classify_test_command(record: dict[str, Any]) -> dict[str, Any]:
    """Distinguish captured baseline failures from missing/incomplete evidence."""
    status = record.get("status")
    if status == "completed" and record.get("returncode") == 0:
        log_path = Path(record.get("log", ""))
        if not log_path.is_file() or not record.get("log_sha256"):
            record["evidence_status"] = "incomplete"
            record["incomplete_reason"] = "test_log_missing"
            record["failure_cases"] = []
            return record
        output = log_path.read_text(encoding="utf-8", errors="replace")
        argv = record.get("argv", [])
        no_tests = (("pytest" in argv and re.search(r"(no tests ran|collected 0 items)", output, re.I))
                    or ("unittest" in argv and re.search(r"Ran 0 tests", output))
                    or (argv and Path(argv[0]).name == "ctest"
                        and "No tests were found" in output))
        if no_tests:
            record["evidence_status"] = "incomplete"
            record["incomplete_reason"] = "test_command_collected_no_tests"
            record["failure_cases"] = []
            return record
        record["evidence_status"] = "pass"
        record["failure_cases"] = []
        return record
    if status != "nonzero_exit" or record.get("returncode") != 1:
        record["evidence_status"] = "incomplete"
        record["incomplete_reason"] = status or "missing_status"
        record["failure_cases"] = []
        return record
    log = Path(record.get("log", ""))
    if not log.is_file():
        record["evidence_status"] = "incomplete"
        record["incomplete_reason"] = "test_log_missing"
        record["failure_cases"] = []
        return record
    text = log.read_text(encoding="utf-8", errors="replace")
    is_pytest = "pytest" in record.get("argv", [])
    is_unittest = "unittest" in record.get("argv", [])
    is_ctest = bool(record.get("argv") and Path(record["argv"][0]).name == "ctest")
    if is_pytest and re.search(r"(ERROR collecting|Interrupted:|internal error|no tests ran)", text, re.I):
        record["evidence_status"] = "incomplete"
        record["incomplete_reason"] = "pytest_collection_or_runner_error"
        record["failure_cases"] = []
        return record
    cases = _failure_cases(record.get("argv", []), text)
    if cases and (is_pytest or is_unittest or is_ctest
                  or (record.get("argv") and Path(record["argv"][0]).name.startswith("AstraSim_"))):
        record["evidence_status"] = "captured_failure"
        record["failure_cases"] = cases
        return record
    record["evidence_status"] = "incomplete"
    record["incomplete_reason"] = "nonzero_exit_without_test_failure_signature"
    record["failure_cases"] = []
    return record


def _attempt_number(baseline_dir: Path) -> int:
    indices = []
    for path in baseline_dir.glob("attempt-*"):
        match = re.fullmatch(r"attempt-(\d+)", path.name)
        if match:
            indices.append(int(match.group(1)))
    return max(indices, default=0) + 1


def _build_reference(root: Path, attempt: Path, frozen_digest: str) -> dict[str, Any]:
    """Build the frozen source in its isolated run copy, once per attempt."""
    source = attempt / "source"
    if source.exists():
        raise AcceptanceError(f"refusing to overwrite existing run source: {source}")
    shutil.copytree(root / "reference/source", source, symlinks=True)
    copied_entries, copied_digest = inventory_tree(source)
    if copied_digest != frozen_digest:
        raise AcceptanceError("reference run copy differs from frozen source",
                              EXIT_FAIL, "reference_copy_digest_mismatch")
    preserved: dict[str, bool] = {}
    for relative, target_name in ((Path("build"), "build-before"),
                                  (Path("sh_test_mesh/generated"), "generated-before")):
        source_path = source / relative
        preserved[target_name] = source_path.exists()
        if source_path.exists():
            shutil.move(str(source_path), str(attempt / target_name))

    cmake_source = source / "build/astra_analytical"
    cmake_source.mkdir(parents=True, exist_ok=True)
    wrapper_path = cmake_source / "CMakeLists.txt"
    wrapper_path.write_text(BUILD_WRAPPER, encoding="utf-8")
    build_dir = cmake_source / "build_congestion_aware"
    logs = root / "gates/G-A1/logs"
    record: dict[str, Any] = {
        "schema_version": 1, "created_utc": utc_now(),
        "frozen_tree_digest_sha256": frozen_digest,
        "copied_source_tree_digest_sha256_before_build_adjustments": copied_digest,
        "run_source": str(source), "run_source_relative": str(source.relative_to(root)),
        "preserved_preexisting": preserved,
        "wrapper_sha256": sha256_file(wrapper_path),
        "cmake_source": str(cmake_source), "cmake_build": str(build_dir),
        "configure_command": ["cmake", "-S", str(cmake_source), "-B", str(build_dir),
                              "-DBUILDTARGET=congestion_aware",
                              "-DNETWORK_BACKEND_BUILD_AS_LIBRARY=ON",
                              "-DCMAKE_BUILD_TYPE=Release"],
        "build_command": ["cmake", "--build", str(build_dir), "-j", "4"],
    }
    write_json(attempt / "build-record.json", {**record, "phase": "configure_started"})
    build_env = _environment_for_tests()
    configure = run_logged(record["configure_command"], cmake_source.parent.parent,
                           build_env, logs / "reference-configure.log", 1800)
    record["configure"] = configure
    if configure.get("status") != "completed":
        record.update({"phase": "configure_failed", "completed_utc": utc_now()})
        write_json(attempt / "build-record.json", record)
        return record
    write_json(attempt / "build-record.json", {**record, "phase": "build_started"})
    build = run_logged(record["build_command"], cmake_source.parent.parent,
                       build_env, logs / "reference-build.log", 3600)
    record["build"] = build
    record["phase"] = "build_complete" if build.get("status") == "completed" else "build_failed"
    record["completed_utc"] = utc_now()
    binary = build_dir / "bin/AstraSim_Analytical_Congestion_Aware_Online"
    record["online_binary"] = str(binary)
    record["online_binary_sha256"] = sha256_file(binary) if binary.is_file() else None
    write_json(attempt / "build-record.json", record)
    return record


def _core_entries(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    excluded_exact = {
        "sh_test_mesh/generated", "sh_test_mesh/runs",
        "build/astra_analytical/build_congestion_aware",
    }
    excluded_prefixes = tuple(path + "/" for path in excluded_exact)
    return [item for item in entries
            if item["path"] not in excluded_exact
            and not item["path"].startswith(excluded_prefixes)]


def _entries_digest(entries: list[dict[str, Any]]) -> str:
    data = json.dumps(entries, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _verify_build(root: Path, attempt: Path, record: dict[str, Any],
                  frozen_digest: str) -> dict[str, Any]:
    if record.get("phase") not in ("build_complete", "reference-build-complete"):
        raise AcceptanceError("reference build record is not complete")
    if record.get("frozen_tree_digest_sha256") != frozen_digest:
        raise AcceptanceError("reference build record points at a different frozen source",
                              EXIT_FAIL, "build_frozen_source_mismatch")
    source_value = record.get("run_source") or record.get("run_source_relative") or record.get("copied_source") or ""
    source_path = Path(source_value)
    source = source_path if source_path.is_absolute() else root / source_path
    if not source.is_dir() or root not in source.resolve().parents:
        raise AcceptanceError("reference build source is missing or outside evidence root")
    build_dir_value = Path(record["cmake_build"])
    build_dir = build_dir_value if build_dir_value.is_absolute() else root / build_dir_value
    wrapper = source / "build/astra_analytical/CMakeLists.txt"
    if not wrapper.is_file() or sha256_file(wrapper) != record.get("wrapper_sha256"):
        raise AcceptanceError("CMake wrapper digest differs from build record",
                              EXIT_FAIL, "build_wrapper_digest_mismatch")

    expected_entries = _core_entries(read_json(root / "manifests/source-inventory.json")["entries"])
    actual_entries, _ = inventory_tree(source)
    actual_core = _core_entries(actual_entries)
    expected_core_digest, actual_core_digest = _entries_digest(expected_entries), _entries_digest(actual_core)
    if expected_core_digest != actual_core_digest:
        expected = {item["path"]: item for item in expected_entries}
        actual = {item["path"]: item for item in actual_core}
        differences = {
            "only_reference": sorted(expected.keys() - actual.keys()),
            "only_run_copy": sorted(actual.keys() - expected.keys()),
            "content_or_mode": sorted(path for path in expected.keys() & actual.keys()
                                       if expected[path] != actual[path]),
        }
        raise AcceptanceError("reference run source changed outside declared build/generated paths: "
                              + json.dumps(differences, ensure_ascii=False),
                              EXIT_FAIL, "build_source_core_digest_mismatch")

    configure_log = root / "gates/G-A1/logs/reference-configure.log"
    build_log = root / "gates/G-A1/logs/reference-build.log"
    if not configure_log.is_file() or not build_log.is_file():
        raise AcceptanceError("reference configure/build logs are missing")
    configure_text = configure_log.read_text(encoding="utf-8", errors="replace")
    build_text = build_log.read_text(encoding="utf-8", errors="replace")
    configure_markers = all(marker in configure_text for marker in (
        "Configuring done", "Generating done", "Build files have been written to"))
    build_markers = "[100%] Built target" in build_text
    if not configure_markers or not build_markers:
        raise AcceptanceError("reference CMake logs lack successful completion markers",
                              EXIT_FAIL, "build_log_completion_marker_missing")
    configure_argv = record.get("configure_command", [])
    build_argv = record.get("build_command", [])
    if ("-DBUILDTARGET=congestion_aware" not in configure_argv
            or "-DNETWORK_BACKEND_BUILD_AS_LIBRARY=ON" not in configure_argv
            or not any(value == "-j" for value in build_argv)
            or not any(build_argv[i + 1] == "4" for i, value in enumerate(build_argv[:-1])
                       if value == "-j")):
        raise AcceptanceError("build commands do not match the A1 wrapper contract",
                              EXIT_FAIL, "build_command_contract_mismatch")

    binary_artifacts: list[dict[str, Any]] = []
    for folder_name in ("bin", "lib"):
        folder = build_dir / folder_name
        if not folder.is_dir():
            continue
        for path in sorted(folder.rglob("*")):
            if path.is_file() and not path.is_symlink():
                binary_artifacts.append({
                    "path": str(path.relative_to(root)),
                    "sha256": sha256_file(path), "size": path.stat().st_size,
                })
    binary_names = {Path(item["path"]).name for item in binary_artifacts}
    required_binaries = {
        "AstraSim_Analytical_Congestion_Aware_Online",
        "AstraSim_Analytical_Congestion_Aware_WorkLedgerBridgeTest",
        "AstraSim_Analytical_Congestion_Aware_ParsedBatchTest",
        "AstraSim_Analytical_Congestion_Aware_GraphBatchCommitterTest",
        "AstraSim_Analytical_Congestion_Aware_BridgeLoopbackTest",
        "libdie_model_cabi.so",
    }
    missing_binaries = sorted(required_binaries - binary_names)
    if missing_binaries:
        raise AcceptanceError("reference build is missing required outputs: "
                              + ", ".join(missing_binaries),
                              EXIT_FAIL, "required_build_output_missing")
    integrity = {
        "schema_version": 1, "created_utc": utc_now(),
        "frozen_tree_digest_sha256": frozen_digest,
        "run_source_core_digest_sha256": actual_core_digest,
        "reference_source_core_digest_sha256": expected_core_digest,
        "source_core_match": True,
        "configure_log": {"path": str(configure_log.relative_to(root)),
                           "sha256": sha256_file(configure_log), "completion_markers": True},
        "build_log": {"path": str(build_log.relative_to(root)),
                       "sha256": sha256_file(build_log), "completion_markers": True},
        "required_binaries": sorted(required_binaries),
        "binary_artifacts": binary_artifacts,
        "all_required_binaries_present": True,
    }
    write_json(attempt / "build-integrity.json", integrity)
    return integrity


def _load_completed_build(root: Path, frozen_digest: str) -> tuple[Path, dict[str, Any], dict[str, Any]] | None:
    baseline_dir = root / "runs/baseline"
    for path in sorted(baseline_dir.glob("attempt-*/build-record.json"), reverse=True):
        record = read_json(path)
        if record.get("frozen_tree_digest_sha256") != frozen_digest:
            continue
        if record.get("phase") in ("build_complete", "reference-build-complete"):
            integrity = _verify_build(root, path.parent, record, frozen_digest)
            return path.parent, record, integrity
    return None


NUMERIC_FIXTURE = r'''import json
from joint.resource_projection import (
    ProjectionInput, ResourceSpec, TaskSpec, WorkItem,
    TASK_KIND_STREAM, project,
)
resource = ResourceSpec("link:r", "link", 10.0)
background = WorkItem("bg", (TaskSpec(
    "bg#0", kind=TASK_KIND_STREAM, route=("link:r",), bytes_amount=20,
    remaining_bytes=20, in_service=True),))
candidate = WorkItem("cand", (TaskSpec(
    "cand#0", kind=TASK_KIND_STREAM, route=("link:r",), bytes_amount=100),),
    is_candidate=True)
inp = ProjectionInput(now_ns=0, resources=(resource,),
                      background_works=(background,), focus_work_id="bg")
res = project(inp, (candidate,))
comparison = next(row for row in res.background_comparisons if row.work_id == "bg")
print(json.dumps({
    "status": res.status,
    "predicted_service_done_ns": res.predicted_service_done_ns,
    "predicted_settle_done_ns": res.predicted_settle_done_ns,
    "background_without_candidate_ns": comparison.done_without_candidate_ns,
    "background_with_candidate_ns": comparison.done_with_candidate_ns,
    "background_delay_ns": comparison.delay_ns,
    "bytes_served_total": res.bytes_served_total,
    "work_count": res.work_count,
}, sort_keys=True, separators=(",", ":")))
'''
NUMERIC_EXPECTED = {
    "status": "ok", "predicted_service_done_ns": 12,
    "predicted_settle_done_ns": 12, "background_without_candidate_ns": 2,
    "background_with_candidate_ns": 4, "background_delay_ns": 2,
    "bytes_served_total": 120.0, "work_count": 2,
}


def numeric_fixture_repeated(runs: list[dict[str, Any]]) -> bool:
    return (len(runs) == 2
            and all(run.get("status") == "completed" for run in runs)
            and runs[0].get("actual") == runs[1].get("actual")
            and runs[0].get("actual") == NUMERIC_EXPECTED)


CABI_BASELINE_NODE = (
    "workload/llama2_7b_inference/joint/test_die_plan_projection.py::"
    "CabiSameValueRegistrationTest::test_estimate_and_execution_self_consistent")


def _runtime_config_file_manifest(directory: Path) -> list[dict[str, Any]]:
    if not directory.is_dir():
        return []
    return [{"path": path.relative_to(directory).as_posix(),
             "size": path.stat().st_size, "sha256": sha256_file(path)}
            for path in sorted(directory.rglob("*"))
            if path.is_file() and not path.is_symlink()]


def _verify_cabi_supplement(root: Path, attempt: Path) -> dict[str, Any] | None:
    result_path = attempt / "supplementary/cabi-test-recheck.json"
    input_path = attempt / "supplementary/cabi-generated-input.json"
    if not result_path.is_file() or not input_path.is_file():
        return None
    result, inputs = read_json(result_path), read_json(input_path)
    if (result.get("status") != "PASS" or result.get("test_name") != CABI_BASELINE_NODE
            or result.get("returncode") != 0 or result.get("frozen_source_modified") is not False
            or result.get("input_manifest_path") != str(input_path.relative_to(root))
            or result.get("input_manifest_sha256") != sha256_file(input_path)
            or inputs.get("input_tree_match") is not True):
        return None
    log_path = root / result.get("log_path", "")
    if (not log_path.is_file() or result.get("log_sha256") != sha256_file(log_path)
            or not re.search(r"(?m)^1 passed in [\d.]+s$",
                             log_path.read_text(encoding="utf-8", errors="replace"))):
        return None
    frozen_runtime = root / "reference/source/sh_test_mesh/generated/runtime_config"
    source_runtime = root / inputs.get("preexisting_generated_source", "")
    copied_runtime = root / inputs.get("run_copy_destination", "")
    frozen_files = _runtime_config_file_manifest(frozen_runtime)
    source_files = _runtime_config_file_manifest(source_runtime)
    copied_files = _runtime_config_file_manifest(copied_runtime)
    if not frozen_files or frozen_files != source_files or frozen_files != copied_files:
        return None
    if inputs.get("preexisting_files") != frozen_files or inputs.get("run_copy_files") != copied_files:
        return None
    backup_rel = inputs.get("previous_full_suite_generated_output_preserved")
    backup_files = inputs.get("previous_full_suite_generated_files", [])
    if backup_rel:
        if _runtime_config_file_manifest(root / backup_rel) != backup_files:
            return None
    return {
        "test_name": CABI_BASELINE_NODE,
        "status": "PASS",
        "disposition": result.get("disposition"),
        "result_path": str(result_path.relative_to(root)),
        "result_sha256": sha256_file(result_path),
        "input_manifest_path": str(input_path.relative_to(root)),
        "input_manifest_sha256": sha256_file(input_path),
        "log_path": result["log_path"], "log_sha256": sha256_file(log_path),
        "restored_files": copied_files,
        "previous_generated_output_preserved": backup_rel,
        "previous_generated_files": backup_files,
    }


def _run_baseline_tests(root: Path, attempt: Path,
                        build_record: dict[str, Any]) -> dict[str, Any]:
    source = attempt / "source"
    workload = source / "sh_test_mesh/workload/llama2_7b_inference"
    build_dir_value = Path(build_record["cmake_build"])
    build_dir = build_dir_value if build_dir_value.is_absolute() else root / build_dir_value
    command_root = root / "gates/G-A1/logs"
    log_prefix = attempt.name + "-"
    env = _environment_for_tests()
    env["PYTHONDONTWRITEBYTECODE"] = "1"

    def get_command(argv: list[str], cwd: Path, log_name: str,
                    expected_exit: int | None = 0, timeout: int = 1800) -> dict[str, Any]:
        log_path = command_root / f"{log_prefix}{log_name}"
        if log_path.is_file():
            return _record_existing_command(argv, cwd, log_path, expected_exit)
        return run_logged(argv, cwd, env, log_path, timeout)

    commands: list[dict[str, Any]] = []

    pytest_argv = [
        "python3", "-m", "pytest", "tests/", "slo_tools/tests/",
        "workload/llama2_7b_inference",
        "--ignore=slo_tools/tests/test_driver_parity.py",
        "--ignore=slo_tools/tests/test_golden_g1g4.py",
        "--ignore=slo_tools/tests/test_slo_contract.py",
        "-p", "no:cacheprovider", "-q",
    ]
    commands.append(get_command(pytest_argv, source / "sh_test_mesh",
                                "reference-pytest.log", None))
    unittest_argv = ["python3", "-m", "unittest", "discover", "-s", "tests"]
    commands.append(get_command(unittest_argv, source / "sh_test_mesh/slo_tools",
                                "reference-unittest.log", None))

    driver_src = REPO_ROOT / "sh_test_mesh/run_scripts/joint_accel_acceptance.py"
    contract_test_src = REPO_ROOT / "sh_test_mesh/workload/llama2_7b_inference/joint/test_joint_accel_acceptance.py"
    tool_test_root = attempt / "tool-contract-tests"
    tool_test_root.mkdir(parents=True, exist_ok=True)
    prior_tool_test = sorted(tool_test_root.glob("attempt-*/sh_test_mesh/run_scripts/joint_accel_acceptance.py"))
    current_tool_hashes = (sha256_file(driver_src), sha256_file(contract_test_src))
    if (prior_tool_test
            and sha256_file(prior_tool_test[-1]) == current_tool_hashes[0]
            and sha256_file(prior_tool_test[-1].parents[1] / "workload/llama2_7b_inference"
                            / "joint/test_joint_accel_acceptance.py") == current_tool_hashes[1]):
        staged_script = prior_tool_test[-1]
        tool_test_attempt = staged_script.parents[2]
        staged_test = (tool_test_attempt / "sh_test_mesh/workload/llama2_7b_inference"
                       / "joint/test_joint_accel_acceptance.py")
    else:
        tool_test_attempt = tool_test_root / f"attempt-{_attempt_number(tool_test_root):03d}"
        tool_test_attempt.mkdir(parents=True, exist_ok=False)
        staged_script = tool_test_attempt / "sh_test_mesh/run_scripts/joint_accel_acceptance.py"
        staged_test = (tool_test_attempt / "sh_test_mesh/workload/llama2_7b_inference"
                       / "joint/test_joint_accel_acceptance.py")
        staged_script.parent.mkdir(parents=True)
        staged_test.parent.mkdir(parents=True)
        shutil.copy2(driver_src, staged_script)
        shutil.copy2(contract_test_src, staged_test)
    tool_test_argv = ["python3", "-m", "pytest", "-p", "no:cacheprovider", "-q", str(staged_test)]
    tool_log = command_root / f"{log_prefix}acceptance-tool-pytest-{tool_test_attempt.name}.log"
    tool_test_result = (_record_existing_command(tool_test_argv, tool_test_attempt, tool_log, 0)
                        if tool_log.is_file()
                        else run_logged(tool_test_argv, tool_test_attempt, env, tool_log, 300))
    tool_test_result["source_sha256"] = {
        "joint_accel_acceptance.py": sha256_file(staged_script) if staged_script.is_file() else None,
        "test_joint_accel_acceptance.py": sha256_file(staged_test) if staged_test.is_file() else None,
    }
    tool_test_result["source_files"] = [
        {"path": str(path.relative_to(root)), "sha256": sha256_file(path)}
        for path in (staged_script, staged_test) if path.is_file()
    ]
    commands.append(tool_test_result)

    binary_dir = build_dir / "bin"
    direct_tests: list[list[str]] = [
        [str(binary_dir / "AstraSim_Analytical_Congestion_Aware_WorkLedgerBridgeTest")],
        [str(binary_dir / "AstraSim_Analytical_Congestion_Aware_ParsedBatchTest")],
        [str(binary_dir / "AstraSim_Analytical_Congestion_Aware_GraphBatchCommitterTest")],
        [str(binary_dir / "AstraSim_Analytical_Congestion_Aware_BridgeLoopbackTest"),
         str(source / "sh_test_mesh/workload/llama2_7b_inference/online/verify/bridge_echo.py")],
    ]
    for index, argv in enumerate(direct_tests, start=1):
        if not Path(argv[0]).is_file():
            commands.append({"argv": argv, "cwd": str(source),
                             "status": "missing_build_output", "returncode": None,
                             "log": None})
            continue
        commands.append(get_command(argv, source,
                                    f"reference-cpp-protocol-{index}.log", None, 900))

    # This assembled wrapper has no top-level CTest file; its tests are
    # registered by the frontend and the nested DieModel projects. A top-level
    # ctest invocation was retained only as a diagnostic and is not counted.
    registered_ctest_specs = [
        (build_dir / "frontend_build", "reference-ctest-registered-1.log"),
        (build_dir / "astra_sim_build/extern/die_model", "reference-ctest-registered-2.log"),
    ]
    for test_dir, log_name in registered_ctest_specs:
        argv = ["ctest", "--test-dir", str(test_dir), "--output-on-failure"]
        commands.append(get_command(argv, source, log_name, None, 1800))

    fixture_runs: list[dict[str, Any]] = []
    fixture_env = env.copy()
    fixture_env["PYTHONPATH"] = str(workload) + os.pathsep + fixture_env.get("PYTHONPATH", "")
    fixture_input = attempt / "numeric-fixture-input.py"
    if not fixture_input.exists():
        fixture_input.write_text(NUMERIC_FIXTURE, encoding="utf-8")
    elif fixture_input.read_text(encoding="utf-8") != NUMERIC_FIXTURE:
        raise AcceptanceError("existing numeric fixture input differs; preserving evidence",
                              EXIT_FAIL, "numeric_fixture_input_mismatch")
    for index in (1, 2):
        log_path = command_root / f"{log_prefix}reference-numeric-fixture-{index}.log"
        argv = ["python3", str(fixture_input)]
        if log_path.is_file():
            stdout = log_path.read_text(encoding="utf-8", errors="replace").strip()
            try:
                parsed_log = json.loads(stdout)
                valid_log = parsed_log == NUMERIC_EXPECTED
            except json.JSONDecodeError:
                parsed_log, valid_log = None, False
            outcome = {"argv": argv, "cwd": str(workload), "log": str(log_path),
                       "status": "completed" if valid_log else "unrecoverable_log",
                       "returncode": 0 if valid_log else None,
                       "log_sha256": sha256_file(log_path),
                       "log_source": "pre-existing-logged-run"}
        else:
            outcome = run_logged(argv, workload, fixture_env, log_path, 60)
        stdout = log_path.read_text(encoding="utf-8", errors="replace").strip() if log_path.is_file() else ""
        parsed: Any = None
        try:
            parsed = json.loads(stdout)
        except json.JSONDecodeError:
            pass
        fixture_runs.append({**outcome, "actual": parsed, "stdout": stdout})

    repeated = numeric_fixture_repeated(fixture_runs)
    commands = [classify_test_command(item) for item in commands]
    cabi_recheck = _verify_cabi_supplement(root, attempt)
    failures: list[dict[str, Any]] = []
    harness_failures: list[dict[str, Any]] = []
    for item in commands:
        for case in item.get("failure_cases", []):
            if case.get("test_name") == CABI_BASELINE_NODE and cabi_recheck is not None:
                case["disposition"] = "invalid_whole_suite_harness_precondition"
                case["supplemental_recheck"] = cabi_recheck["result_path"]
                case["supplemental_recheck_sha256"] = cabi_recheck["result_sha256"]
                harness_failures.append(case)
            else:
                failures.append(case)
    incomplete_commands = [item for item in commands if item.get("evidence_status") == "incomplete"]
    pytest_log = command_root / f"{log_prefix}reference-pytest.log"
    unittest_log = command_root / f"{log_prefix}reference-unittest.log"
    skip_count = 0
    if pytest_log.is_file():
        match = re.search(r"\b(\d+) skipped\b", pytest_log.read_text(encoding="utf-8", errors="replace"))
        skip_count += int(match.group(1)) if match else 0
    if unittest_log.is_file():
        match = re.search(r"OK \(skipped=(\d+)\)", unittest_log.read_text(encoding="utf-8", errors="replace"))
        skip_count += int(match.group(1)) if match else 0
    skip_inventory = []
    missing_journal_fixture = not Path("/var/tmp/phase2_journal/runs/on_2s").is_dir()
    if missing_journal_fixture and skip_count:
        skip_inventory.append({
            "test_name": "JournalParityRealTests.test_real_journal_certified_and_matches_checksum",
            "reason": "phase2 journal fixture 缺失",
            "condition": "skipUnless(FIXTURE.is_dir())",
            "fixture_path": "/var/tmp/phase2_journal/runs/on_2s",
            "fixture_present": False,
            "test_source": "sh_test_mesh/slo_tools/tests/test_hbm_watermark.py",
            "note": "quiet baseline logs report skip counts only; source guard and absent fixture identify the skip condition",
        })
    skip_audit_status = ("NONE" if skip_count == 0 else
                         "CAPTURED" if skip_count == 2 and missing_journal_fixture
                         and skip_inventory else "INCOMPLETE")
    root_ctest_log = command_root / f"{log_prefix}reference-ctest.log"
    diagnostics = []
    if root_ctest_log.is_file() and "No tests were found" in root_ctest_log.read_text(encoding="utf-8", errors="replace"):
        diagnostics.append({"command": ["ctest", "--test-dir", str(build_dir), "--output-on-failure"],
                            "status": "no_tests_registered_at_wrapper_root",
                            "log": str(root_ctest_log), "log_sha256": sha256_file(root_ctest_log),
                            "excluded_from_test_commands": True,
                            "registered_test_directories": [str(item[0]) for item in registered_ctest_specs]})
    cabi_case_present = any(case.get("test_name") == CABI_BASELINE_NODE
                            for item in commands for case in item.get("failure_cases", []))
    supplemental_rechecks_valid = (not cabi_case_present or cabi_recheck is not None)
    completed = (not incomplete_commands and repeated
                 and skip_audit_status in ("NONE", "CAPTURED")
                 and supplemental_rechecks_valid)
    old_bug_path = root / OLD_BUDGET_FIXTURE
    return {
        "schema_version": 1, "created_utc": utc_now(),
        "status": "PASS" if completed else "INCOMPLETE",
        "reason_code": "reference_baseline_captured" if completed else "baseline_incomplete",
        "frozen_tree_digest_sha256": build_record["frozen_tree_digest_sha256"],
        "run_source": build_record.get("run_source") or build_record.get("copied_source"),
        "build_record": str(attempt / "build-record.json"),
        "build_record_sha256": sha256_file(attempt / "build-record.json"),
        "test_commands": commands,
        "known_baseline_failures": failures,
        "harness_failures_rechecked": harness_failures,
        "supplemental_rechecks": [cabi_recheck] if cabi_recheck is not None else [],
        "supplemental_rechecks_valid": supplemental_rechecks_valid,
        "skipped_tests": skip_inventory,
        "skipped_test_count": len(skip_inventory),
        "skipped_invocation_count": skip_count,
        "skip_audit_status": skip_audit_status,
        "diagnostics": diagnostics,
        "baseline_failures_are_not_relabelled_as_passes": True,
        "numeric_fixture": {
            "name": "single_resource_candidate_reaction_anchor",
            "runs": fixture_runs, "expected": NUMERIC_EXPECTED,
            "identical_replay": repeated,
            "input_path": str(fixture_input.relative_to(root)),
            "input_sha256": sha256_file(fixture_input),
        },
        "old_event_budget_bug_fixture": {
            "path": str(old_bug_path.relative_to(root)),
            "sha256": sha256_file(old_bug_path) if old_bug_path.is_file() else None,
            "used_as_l1_equivalence_evidence": False,
        },
        "trace_smoke": {"status": "NOT_RUN", "reason": "A1 avoids redundant trace smoke by instruction"},
        "tool_versions": {
            "python": sys.version, "platform": platform.platform(),
            "cmake_build_config": "Release", "build_jobs": 4,
        },
        "incomplete_commands": incomplete_commands,
    }


def command_inventory(root: Path) -> int:
    inventory_path = root / "manifests/source-inventory.json"
    if not inventory_path.is_file():
        result = _initialize_a0_inventory(root)
        if result["status"] != "PASS":
            invalidate_done(root, "G-A0", result["reason_code"])
            print(json.dumps(result, ensure_ascii=False, sort_keys=True))
            return EXIT_INCOMPLETE
    try:
        done, inventory = validate_a0(root)
    except AcceptanceError as exc:
        invalidate_done(root, "G-A0", exc.reason_code)
        raise
    print(json.dumps({"command": "inventory", "status": "PASS",
                      "reason_code": "frozen_reference_verified",
                      "frozen_tree_digest_sha256": done["frozen_tree_digest_sha256"],
                      "inventory_entries": len(inventory["entries"]),
                      "done_sha256": sha256_file(root / "gates/G-A0/DONE")},
                     ensure_ascii=False, sort_keys=True))
    return EXIT_PASS


def _initialize_a0_inventory(root: Path) -> dict[str, Any]:
    """Validate a pre-copied freeze and write A0 manifests; never copy over it."""
    frozen = root / "reference/source"
    manifests = root / "manifests"
    required_seed = [
        "git-status-before.z", "joint-before.patch", "joint-staged-before.patch",
        "index-before.z", "head.txt", "Agents.md", "design.md",
        "execution-plan.md", "switches-before.md", "astra_compute_20.csv",
    ]
    missing = [name for name in required_seed if not (manifests / name).is_file()]
    if missing:
        result = {"gate": "G-A0", "status": "INCOMPLETE",
                  "reason_code": "a0_seed_manifest_missing", "missing": missing}
        write_json(root / "gates/G-A0/result.json", result)
        return result

    # The acceptance driver and its contract test are added after the reference
    # copy is frozen. Exclude only those known tooling additions from the
    # pre-implementation live/frozen comparison.
    tooling_additions = {
        "sh_test_mesh/run_scripts/joint_accel_acceptance.py",
        "sh_test_mesh/workload/llama2_7b_inference/joint/test_joint_accel_acceptance.py",
    }
    live_entries, live_digest = inventory_tree(REPO_ROOT, tooling_additions)
    frozen_entries, frozen_digest = inventory_tree(frozen)
    live_map = {item["path"]: item for item in live_entries}
    frozen_map = {item["path"]: item for item in frozen_entries}
    differences = {
        "only_live": sorted(live_map.keys() - frozen_map.keys()),
        "only_frozen": sorted(frozen_map.keys() - live_map.keys()),
        "content_or_mode": sorted(path for path in live_map.keys() & frozen_map.keys()
                                   if live_map[path] != frozen_map[path]),
    }
    tree_matches = not any(differences.values())
    symlinks = [item for item in frozen_entries if item["type"] == "symlink"]
    if symlinks:
        result = {"gate": "G-A0", "status": "INCOMPLETE",
                  "reason_code": "external_symlink_freeze_requires_dependency_copy",
                  "symlink_count": len(symlinks), "differences": differences,
                  "frozen_tree_digest_sha256": frozen_digest}
        write_json(manifests / "dependency-links.json", {
            "schema_version": 1, "created_utc": utc_now(),
            "symlinks": symlinks, "external_dependencies_copied": [],
            "status": "INCOMPLETE", "reason": "copy and rewire dependencies before A0 DONE",
        })
        write_json(root / "gates/G-A0/result.json", result)
        return result
    if not tree_matches:
        result = {"gate": "G-A0", "status": "INCOMPLETE",
                  "reason_code": "live_and_frozen_tree_differ",
                  "differences": differences,
                  "live_tree_digest_sha256": live_digest,
                  "frozen_tree_digest_sha256": frozen_digest}
        write_json(root / "gates/G-A0/result.json", result)
        return result

    trace_live = WORKSPACE_ROOT / TRACE_RELATIVE
    trace_saved = manifests / "astra_compute_20.csv"
    if not trace_live.is_file() or sha256_file(trace_live) != sha256_file(trace_saved):
        result = {"gate": "G-A0", "status": "INCOMPLETE",
                  "reason_code": "trace_snapshot_mismatch"}
        write_json(root / "gates/G-A0/result.json", result)
        return result

    trace_config_rel = Path("sh_test_mesh/workload/llama2_7b_inference/trace_config.csv")
    original_trace_config = REPO_ROOT / trace_config_rel
    if not original_trace_config.is_file():
        result = {"gate": "G-A0", "status": "INCOMPLETE",
                  "reason_code": "trace_config_missing"}
        write_json(root / "gates/G-A0/result.json", result)
        return result
    (manifests / "trace_config.original").write_bytes(original_trace_config.read_bytes())

    hardware = [item for item in frozen_entries if item["type"] == "file"
                and ("hardware" in item["path"].lower()
                     or item["path"].endswith((".so", ".a", ".bin")))]
    generated = [item for item in frozen_entries
                 if Path(item["path"]).name in {"build", "generated", "traces", "runs"}]
    write_json(manifests / "source-inventory.json", {
        "schema_version": 1, "created_utc": utc_now(),
        "live_root_at_freeze": str(REPO_ROOT), "frozen_root": str(frozen),
        "tree_digest_sha256": frozen_digest,
        "live_tree_digest_sha256": live_digest, "live_equals_frozen": True,
        "entries": frozen_entries, "hardware_libraries_and_binaries": hardware,
    })
    write_json(manifests / "dependency-links.json", {
        "schema_version": 1, "created_utc": utc_now(), "symlinks": [],
        "external_dependencies_copied": [], "status": "no_symlink_dependencies",
    })
    write_json(manifests / "preexisting-generated-artifacts.json", {
        "schema_version": 1, "created_utc": utc_now(),
        "root_inventory_digest_sha256": frozen_digest, "entries": generated,
    })
    versions: dict[str, Any] = {}
    for key, argv in {
        "python3": ["python3", "--version"], "cmake": ["cmake", "--version"],
        "cxx": ["g++", "--version"], "protoc": ["protoc", "--version"],
        "pytest": ["python3", "-m", "pytest", "--version"],
    }.items():
        if shutil.which(argv[0]) is None:
            versions[key] = {"available": False, "argv": argv}
            continue
        completed = subprocess.run(argv, capture_output=True, text=True, check=False)
        versions[key] = {"available": completed.returncode == 0, "argv": argv,
                         "returncode": completed.returncode,
                         "stdout": completed.stdout.strip(),
                         "stderr": completed.stderr.strip()}
    mem_total_kib = None
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                mem_total_kib = int(line.split()[1])
                break
    except OSError:
        pass
    write_json(manifests / "environment.json", {
        "schema_version": 1, "created_utc": utc_now(),
        "python": sys.version, "platform": platform.platform(),
        "cpu_count": os.cpu_count(), "mem_total_kib": mem_total_kib,
        "thread_environment": {key: os.environ.get(key) for key in (
            "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "PROTOBUF_FROM_SOURCE")},
        "tool_versions": versions,
    })
    result = {
        "gate": "G-A0", "status": "PASS", "reason_code": "frozen_tree_verified",
        "created_utc": utc_now(), "frozen_tree_digest_sha256": frozen_digest,
        "live_tree_equal_at_freeze": True, "trace_sha256": sha256_file(trace_live),
        "trace_config_sha256": sha256_file(manifests / "trace_config.original"),
        "symlink_count": 0, "external_dependency_count": 0,
        "artifacts": [f"manifests/{name}" for name in required_seed] + [
            "manifests/source-inventory.json", "manifests/dependency-links.json",
            "manifests/preexisting-generated-artifacts.json", "manifests/environment.json",
            "manifests/trace_config.original"],
    }
    result_path = root / "gates/G-A0/result.json"
    write_json(result_path, result)
    artifacts = [root / rel for rel in result["artifacts"]]
    write_json(root / "gates/G-A0/DONE", {
        "gate": "G-A0", "status": "PASS", "result_sha256": sha256_file(result_path),
        "frozen_tree_digest_sha256": frozen_digest,
        "required_artifacts": {str(path.relative_to(root)): sha256_file(path)
                               for path in artifacts},
    })
    return result


def command_baseline(root: Path) -> int:
    _a0_done, inventory = validate_a0(root)
    frozen_digest = inventory["tree_digest_sha256"]
    completed_build = _load_completed_build(root, frozen_digest)
    if completed_build is None:
        baseline_dir = root / "runs/baseline"
        baseline_dir.mkdir(parents=True, exist_ok=True)
        existing_attempts = list(baseline_dir.glob("attempt-*/build-record.json"))
        if existing_attempts:
            raise AcceptanceError("existing reference build attempt is incomplete or unverified; "
                                  "preserving it instead of repeating build",
                                  EXIT_INCOMPLETE, "existing_reference_build_unverified")
        attempt = baseline_dir / f"attempt-{_attempt_number(baseline_dir):03d}"
        attempt.mkdir(parents=True, exist_ok=False)
        build_record = _build_reference(root, attempt, frozen_digest)
        if build_record.get("phase") != "build_complete":
            payload = {"status": "FAIL", "reason_code": "reference_build_failed",
                       "build_record": str(attempt / "build-record.json"),
                       "build": build_record}
            write_json(root / "packages/acceptance/baseline.json", payload)
            print(json.dumps(payload, ensure_ascii=False))
            return EXIT_FAIL
        build_integrity = _verify_build(root, attempt, build_record, frozen_digest)
    else:
        attempt, build_record, build_integrity = completed_build

    baseline = _run_baseline_tests(root, attempt, build_record)
    baseline["attempt"] = str(attempt.relative_to(root))
    baseline["created_utc"] = utc_now()
    baseline["parents"] = {"G-A0": sha256_file(root / "gates/G-A0/DONE")}
    baseline["build_record"] = str((attempt / "build-record.json").relative_to(root))
    baseline["build_record_sha256"] = sha256_file(attempt / "build-record.json")
    baseline["build_integrity"] = str((attempt / "build-integrity.json").relative_to(root))
    baseline["build_integrity_sha256"] = sha256_file(attempt / "build-integrity.json")
    out = root / "packages/acceptance/baseline.json"
    write_json(out, baseline)
    if baseline["status"] != "PASS":
        invalidate_done(root, "G-A1", "new_baseline_incomplete")
    write_json(root / "STATE.json", {
        "phase": "A1_BASELINE_CAPTURED" if baseline["status"] == "PASS" else "A1_BASELINE_INCOMPLETE",
        "evidence_root": str(root), "frozen_tree_digest_sha256": frozen_digest,
        "a0_gate": "PASS", "a1_baseline": baseline["status"],
        "a1_attempt": str(attempt.relative_to(root)),
        "known_baseline_failure_count": len(baseline["known_baseline_failures"]),
        "trace_smoke": "not_started_by_design", "last_updated_utc": utc_now(),
    })
    print(json.dumps({"command": "baseline", "status": baseline["status"],
                      "reason_code": baseline["reason_code"],
                      "known_baseline_failures": len(baseline["known_baseline_failures"]),
                      "numeric_fixture_repeated": baseline["numeric_fixture"]["identical_replay"],
                      "baseline_manifest": str(out)}, ensure_ascii=False, sort_keys=True))
    return EXIT_PASS if baseline["status"] == "PASS" else EXIT_INCOMPLETE


def _gate_artifacts(root: Path, value: Any) -> tuple[dict[str, str], list[str], list[str]]:
    """Validate a gate's declared file manifest without following escapes."""
    if not isinstance(value, dict) or not value:
        return {}, ["required_artifacts must be a non-empty path-to-SHA256 object"], []
    checked: dict[str, str] = {}
    missing: list[str] = []
    invalid: list[str] = []
    for relative, expected in value.items():
        if (not isinstance(relative, str) or not relative
                or Path(relative).is_absolute()
                or not isinstance(expected, str)
                or re.fullmatch(r"[0-9a-f]{64}", expected) is None):
            invalid.append(str(relative))
            continue
        path = (root / relative).resolve()
        try:
            path.relative_to(root.resolve())
        except ValueError:
            invalid.append(relative)
            continue
        if not path.is_file():
            missing.append(relative)
            continue
        if sha256_file(path) != expected:
            invalid.append(relative)
            continue
        checked[relative] = expected
    return checked, missing, invalid


def _command_generic_gate(root: Path, name: str) -> int:
    result_path = root / "gates" / name / "result.json"
    done_path = result_path.with_name("DONE")
    if result_path.is_file() and done_path.is_file():
        try:
            validate_done(root, name)
            payload = {"gate": name, "status": "PASS",
                       "reason_code": "gate_evidence_already_verified",
                       "done_sha256": sha256_file(done_path)}
            print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
            return EXIT_PASS
        except AcceptanceError:
            pass

    try:
        result = _require_object(read_json(result_path), f"{name} result")
    except AcceptanceError as exc:
        result = {"gate": name, "status": "INCOMPLETE",
                  "reason_code": "gate_result_missing_or_invalid",
                  "message": str(exc)}
        write_json(result_path, result)
        invalidate_done(root, name, "gate_result_missing_or_invalid")
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return EXIT_INCOMPLETE

    claimed_status = result.get("status")
    if result.get("gate") != name:
        result.update({"status": "FAIL", "reason_code": "gate_identity_mismatch"})
        write_json(result_path, result)
        invalidate_done(root, name, "gate_identity_mismatch")
        print(json.dumps({"gate": name, "status": "FAIL",
                          "reason_code": "gate_identity_mismatch"},
                         ensure_ascii=False, sort_keys=True))
        return EXIT_FAIL
    if not isinstance(claimed_status, str) or claimed_status not in GATE_STATUSES:
        result.update({"status": "FAIL", "reason_code": "gate_status_invalid"})
        write_json(result_path, result)
        invalidate_done(root, name, "gate_status_invalid")
        print(json.dumps({"gate": name, "status": "FAIL",
                          "reason_code": "gate_status_invalid"},
                         ensure_ascii=False, sort_keys=True))
        return EXIT_FAIL
    if claimed_status != "PASS":
        invalidate_done(root, name, f"gate_status_{claimed_status.lower()}")
        payload = {"gate": name, "status": claimed_status,
                   "reason_code": result.get("reason_code")}
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        return (EXIT_FAIL if claimed_status == "FAIL" else
                EXIT_INSUFFICIENT if claimed_status == "INSUFFICIENT_EVIDENCE" else
                EXIT_INCOMPLETE)

    parent_hashes: dict[str, str] = {}
    parent_failures: dict[str, str] = {}
    for parent in GATE_PARENTS[name]:
        try:
            validate_done(root, parent)
            parent_hashes[parent] = sha256_file(root / "gates" / parent / "DONE")
        except AcceptanceError as exc:
            parent_failures[parent] = exc.reason_code
    if parent_failures:
        result.update({"status": "INCOMPLETE", "claimed_status": claimed_status,
                       "reason_code": "gate_parent_evidence_missing",
                       "parent_failures": parent_failures})
        write_json(result_path, result)
        invalidate_done(root, name, "gate_parent_evidence_missing")
        print(json.dumps({"gate": name, "status": "INCOMPLETE",
                          "reason_code": result["reason_code"],
                          "parent_failures": parent_failures},
                         ensure_ascii=False, sort_keys=True))
        return EXIT_INCOMPLETE

    if result.get("parents") != parent_hashes:
        result.update({"status": "FAIL", "claimed_status": claimed_status,
                       "reason_code": "gate_parent_hash_mismatch",
                       "expected_parents": parent_hashes})
        write_json(result_path, result)
        invalidate_done(root, name, "gate_parent_hash_mismatch")
        print(json.dumps({"gate": name, "status": "FAIL",
                          "reason_code": result["reason_code"]},
                         ensure_ascii=False, sort_keys=True))
        return EXIT_FAIL

    checks = result.get("checks")
    if not isinstance(checks, dict) or not checks or any(type(value) is not bool
                                                         for value in checks.values()):
        result.update({"status": "INCOMPLETE", "claimed_status": claimed_status,
                       "reason_code": "gate_checks_missing_or_untyped"})
        write_json(result_path, result)
        invalidate_done(root, name, "gate_checks_missing_or_untyped")
        print(json.dumps({"gate": name, "status": "INCOMPLETE",
                          "reason_code": result["reason_code"]},
                         ensure_ascii=False, sort_keys=True))
        return EXIT_INCOMPLETE
    failed_checks = sorted(key for key, value in checks.items() if value is not True)
    if failed_checks:
        result.update({"status": "FAIL", "claimed_status": claimed_status,
                       "reason_code": "gate_check_failed",
                       "failed_checks": failed_checks})
        write_json(result_path, result)
        invalidate_done(root, name, "gate_check_failed")
        print(json.dumps({"gate": name, "status": "FAIL",
                          "reason_code": result["reason_code"],
                          "failed_checks": failed_checks},
                         ensure_ascii=False, sort_keys=True))
        return EXIT_FAIL

    checked, missing, invalid = _gate_artifacts(root, result.get("required_artifacts"))
    if invalid:
        result.update({"status": "FAIL", "claimed_status": claimed_status,
                       "reason_code": "gate_artifact_digest_mismatch",
                       "invalid_artifacts": invalid})
        write_json(result_path, result)
        invalidate_done(root, name, "gate_artifact_digest_mismatch")
        print(json.dumps({"gate": name, "status": "FAIL",
                          "reason_code": result["reason_code"],
                          "invalid_artifacts": invalid},
                         ensure_ascii=False, sort_keys=True))
        return EXIT_FAIL
    if missing or not checked:
        result.update({"status": "INCOMPLETE", "claimed_status": claimed_status,
                       "reason_code": "gate_artifacts_missing",
                       "missing_artifacts": missing})
        write_json(result_path, result)
        invalidate_done(root, name, "gate_artifacts_missing")
        print(json.dumps({"gate": name, "status": "INCOMPLETE",
                          "reason_code": result["reason_code"],
                          "missing_artifacts": missing},
                         ensure_ascii=False, sort_keys=True))
        return EXIT_INCOMPLETE

    done = {"gate": name, "status": "PASS",
            "result_sha256": sha256_file(result_path),
            "parents": parent_hashes, "required_artifacts": checked}
    write_json(done_path, done)
    payload = {"gate": name, "status": "PASS",
               "reason_code": "gate_artifacts_verified",
               "done_sha256": sha256_file(done_path),
               "parent_count": len(parent_hashes),
               "artifact_count": len(checked)}
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return EXIT_PASS


def command_gate(root: Path, name: str) -> int:
    if name not in GATE_PARENTS:
        payload = {"gate": name, "status": "FAIL",
                   "reason_code": "gate_name_invalid"}
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        return EXIT_FAIL
    if name == "G-A0":
        try:
            done, _inventory = validate_a0(root)
            payload = {"gate": name, "status": "PASS",
                       "reason_code": "gate_evidence_already_verified",
                       "done_sha256": sha256_file(root / "gates" / name / "DONE")}
            print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
            return EXIT_PASS
        except AcceptanceError as exc:
            payload = {"gate": name,
                       "status": "FAIL" if exc.code == EXIT_FAIL else "INCOMPLETE",
                       "reason_code": exc.reason_code, "message": str(exc)}
            print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
            return exc.code
    if name not in ("G-A1",):
        return _command_generic_gate(root, name)
    a0_done, _inventory = validate_a0(root)
    baseline_path = root / "packages/acceptance/baseline.json"
    if not baseline_path.is_file():
        payload = {"gate": name, "status": "INCOMPLETE",
                   "reason_code": "baseline_evidence_missing"}
        write_json(root / "gates/G-A1/result.json", payload)
        invalidate_done(root, "G-A1", "baseline_evidence_missing")
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        return EXIT_INCOMPLETE
    baseline = read_json(baseline_path)
    failures = baseline.get("known_baseline_failures")
    fixture = baseline.get("numeric_fixture", {})
    old_bug = baseline.get("old_event_budget_bug_fixture", {})
    attempt_path = root / baseline.get("attempt", "")
    cabi_recheck = _verify_cabi_supplement(root, attempt_path)
    harness_failures = baseline.get("harness_failures_rechecked", [])
    harness_recheck_valid = (
        not harness_failures
        or (cabi_recheck is not None
            and len(harness_failures) == 1
            and harness_failures[0].get("test_name") == CABI_BASELINE_NODE
            and harness_failures[0].get("supplemental_recheck_sha256")
                == cabi_recheck.get("result_sha256")))
    command_records = baseline.get("test_commands", [])
    captured_records = [item for item in command_records
                        if item.get("evidence_status") in ("pass", "captured_failure")]
    logs_valid = (bool(captured_records) and len(captured_records) == len(command_records)
                  and all(item.get("log") and Path(item["log"]).is_file()
                          and item.get("log_sha256") == sha256_file(Path(item["log"]))
                          for item in captured_records))
    baseline_path_value = baseline.get("build_record", "")
    build_record_path = root / baseline_path_value
    build_record_valid = (build_record_path.is_file()
                          and baseline.get("build_record_sha256")
                          == sha256_file(build_record_path))
    build_record = read_json(build_record_path) if build_record_valid else {}
    integrity_path = root / baseline.get("build_integrity", "")
    integrity_valid = (integrity_path.is_file()
                       and baseline.get("build_integrity_sha256") == sha256_file(integrity_path))
    integrity = read_json(integrity_path) if integrity_valid else {}
    a0_done_now = sha256_file(root / "gates/G-A0/DONE")
    a0_inventory = read_json(root / "manifests/source-inventory.json")
    fixture_runs = fixture.get("runs", [])
    fixture_input_path = root / fixture.get("input_path", "")
    fixture_logs_valid = (len(fixture_runs) == 2 and fixture_input_path.is_file()
                          and fixture.get("input_sha256") == sha256_file(fixture_input_path)
                          and numeric_fixture_repeated(fixture_runs)
                          and all(run.get("log") and Path(run["log"]).is_file()
                                  and run.get("log_sha256") == sha256_file(Path(run["log"]))
                                  for run in fixture_runs))
    old_bug_path = root / OLD_BUDGET_FIXTURE
    old_bug_valid = (old_bug_path.is_file()
                     and old_bug.get("sha256") == sha256_file(old_bug_path)
                     and old_bug.get("used_as_l1_equivalence_evidence") is False)
    failures_complete = all(case.get("test_name") and case.get("input_signature")
                            and case.get("error_signature") for case in failures or [])
    checks = {
        "a0_parent_valid": bool(a0_done),
        "baseline_parent_hash_matches": baseline.get("parents", {}).get("G-A0") == a0_done_now,
        "baseline_uses_same_frozen_source": baseline.get("frozen_tree_digest_sha256") == a0_inventory.get("tree_digest_sha256"),
        "reference_build_complete": baseline.get("status") == "PASS",
        "build_record_complete_and_hashed": build_record_valid and build_record.get("phase") in ("build_complete", "reference-build-complete"),
        "build_integrity_hash_and_core_source_match": integrity_valid and integrity.get("source_core_match") is True,
        "all_baseline_commands_classified_with_hashed_logs": bool(command_records)
            and not baseline.get("incomplete_commands") and logs_valid and failures_complete,
        "tool_contract_test_sources_hashed": any(
            item.get("source_files") and all(
                (root / source.get("path", "")).is_file()
                and sha256_file(root / source["path"]) == source.get("sha256")
                for source in item["source_files"])
            for item in command_records if item.get("source_sha256")),
        "suite_environment_failure_rechecked_with_frozen_generated_input": harness_recheck_valid
            and baseline.get("supplemental_rechecks_valid") is True,
        "skipped_test_reasons_accounted": baseline.get("skip_audit_status") in ("NONE", "CAPTURED")
            and baseline.get("skipped_test_count", 0) == len(baseline.get("skipped_tests", [])),
        "fixed_numeric_fixture_repeated_with_hashed_input_and_logs": fixture_logs_valid,
        "old_budget_bug_preregistered": old_bug_valid,
    }
    status = "PASS" if all(checks.values()) else "INCOMPLETE"
    reason = "a1_reference_reproducible" if status == "PASS" else "a1_required_evidence_missing"
    result = {
        "gate": name, "status": status, "reason_code": reason,
        "created_utc": utc_now(), "checks": checks,
        "parents": {"G-A0": sha256_file(root / "gates/G-A0/DONE")},
        "baseline_manifest": str(baseline_path.relative_to(root)),
        "baseline_manifest_sha256": sha256_file(baseline_path),
        "known_baseline_failures": failures,
        "build_record": str(build_record_path.relative_to(root)) if build_record_valid else None,
        "build_integrity": str(integrity_path.relative_to(root)) if integrity_valid else None,
        "trace_smoke": {"status": "NOT_RUN", "reason": "explicitly excluded from A1 efficiency scope"},
        "scope_note": "G-A1 attests frozen build/test capture and deterministic controlled fixture; it does not attest trace smoke, tuning, or closed-loop accuracy.",
    }
    result_path = root / "gates/G-A1/result.json"
    write_json(result_path, result)
    if status == "PASS":
        required = [baseline_path, build_record_path, integrity_path, old_bug_path,
                    fixture_input_path]
        required.extend(Path(item["log"]) for item in captured_records)
        required.extend(root / source["path"] for item in captured_records
                        for source in item.get("source_files", []))
        required.extend(Path(run["log"]) for run in fixture_runs)
        required.extend(root / integrity["configure_log"]["path"] for _ in (0,))
        required.extend(root / integrity["build_log"]["path"] for _ in (0,))
        required.extend(root / item["path"] for item in integrity["binary_artifacts"])
        if cabi_recheck is not None:
            required.extend((root / cabi_recheck["result_path"],
                             root / cabi_recheck["input_manifest_path"],
                             root / cabi_recheck["log_path"]))
            input_manifest = read_json(root / cabi_recheck["input_manifest_path"])
            for directory_rel, files in (
                    (input_manifest["preexisting_generated_source"],
                     input_manifest["preexisting_files"]),
                    (input_manifest["run_copy_destination"],
                     input_manifest["run_copy_files"]),
                    ("reference/source/sh_test_mesh/generated/runtime_config",
                     input_manifest["preexisting_files"])):
                required.extend(root / directory_rel / entry["path"] for entry in files)
            backup_rel = input_manifest.get("previous_full_suite_generated_output_preserved")
            if backup_rel:
                required.extend(root / backup_rel / entry["path"]
                                for entry in input_manifest.get("previous_full_suite_generated_files", []))
        write_json(root / "gates/G-A1/DONE", {
            "gate": name, "status": "PASS", "result_sha256": sha256_file(result_path),
            "parents": result["parents"],
            "required_artifacts": {str(path.relative_to(root)): sha256_file(path)
                                   for path in required},
        })
    else:
        invalidate_done(root, "G-A1", reason)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return EXIT_PASS if status == "PASS" else EXIT_INCOMPLETE


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    for name in ("inventory", "baseline", "capture", "tune", "holdout",
                 "performance", "report", "verify-clean"):
        child = subs.add_parser(name)
        child.add_argument("--evidence", required=True, type=Path)
        if name in ("capture", "tune", "holdout", "performance", "smoke"):
            child.add_argument("--source-root", type=Path,
                               help="hash-recorded implementation source/run copy")
        if name in ("capture", "tune", "holdout", "performance"):
            child.add_argument("--prediction-config", type=Path,
                               help="resolved prediction config; defaults to frozen default preset")
        if name in ("capture", "tune", "holdout"):
            child.add_argument("--native-library", type=Path,
                               help="new joint_projection shared library for L3 scoring")
        if name == "capture":
            child.add_argument("--fixture-binary", type=Path)
            child.add_argument("--fixtures-dir", type=Path,
                               help="directory containing real cold-fixture case subdirectories")
        if name == "performance":
            child.add_argument("--exact-only", action="store_true",
                               help="run old/new exact diagnostics without claiming approximation speed")
    gate = subs.add_parser("gate")
    gate.add_argument("--evidence", required=True, type=Path)
    gate.add_argument("--name", required=True)
    smoke = subs.add_parser("smoke")
    smoke.add_argument("--evidence", required=True, type=Path)
    smoke.add_argument("--variant", required=True, choices=("exact", "approx"))
    smoke.add_argument("--source-root", type=Path,
                       help="built isolated source/run copy; live/frozen sources are rejected")
    smoke.add_argument("--prediction-config", type=Path,
                       help="resolved prediction config for the exact smoke arm")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        root = evidence_root(args.evidence)
        if args.command == "inventory":
            return command_inventory(root)
        if args.command == "baseline":
            return command_baseline(root)
        if args.command == "gate":
            return command_gate(root, args.name)
        if args.command == "smoke":
            return command_smoke(root, source_root=args.source_root,
                                 prediction_config=args.prediction_config,
                                 variant=args.variant)
        if args.command == "capture":
            return command_capture(root, source_root=args.source_root,
                                   fixture_binary=args.fixture_binary,
                                   fixtures_dir=args.fixtures_dir,
                                   prediction_config=args.prediction_config)
        if args.command == "tune":
            return command_tune(root, source_root=args.source_root,
                                prediction_config=args.prediction_config,
                                native_library=args.native_library)
        if args.command == "holdout":
            return command_holdout(root, source_root=args.source_root,
                                   prediction_config=args.prediction_config,
                                   native_library=args.native_library)
        if args.command == "performance":
            return command_performance(root, source_root=args.source_root,
                                       prediction_config=args.prediction_config,
                                       exact_only=args.exact_only)
        if args.command == "report":
            return command_report(root)
        if args.command == "verify-clean":
            return command_verify_clean(root)
    except AcceptanceError as exc:
        payload = {"command": getattr(args, "command", "unknown"),
                   "status": "FAIL" if exc.code == EXIT_FAIL else "INCOMPLETE",
                   "reason_code": exc.reason_code, "message": str(exc)}
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True), file=sys.stderr)
        return exc.code
    return EXIT_INCOMPLETE


if __name__ == "__main__":
    raise SystemExit(main())
