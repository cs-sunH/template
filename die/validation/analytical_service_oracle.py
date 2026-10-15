#!/usr/bin/env python3
"""Compare an actual compiled OS Runtime run with the independent K-wave oracle.

This is an offline model-to-model check. The C++ driver supplies the actual
compiled Plan and Runtime observations; the grouped discrete oracle supplies
the reference schedule. A result describes this observed fixture only and is
not a broad granularity certificate.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import compiled_gemm_oracle as reference
import grouped_gemm_oracle as grouped


class UnsupportedScenario(ValueError):
    """The discrete reference's declared fixture coverage does not apply."""


PRECISION_POLICY_ID = "die_accuracy_target_10pct_fallback_20pct_v1"
TARGET_ERROR_LIMIT = 0.10
FALLBACK_ERROR_LIMIT = 0.20


def _normalize_tolerance_evidence(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("tolerance evidence must be a JSON object")
    source = value.get("source")
    attempts = value.get("attempts")
    reason = value.get("reason")
    if not isinstance(source, str) or not source.strip():
        raise ValueError("tolerance evidence requires a nonempty source")
    if (not isinstance(attempts, list) or len(attempts) < 2
            or any(not isinstance(item, str) or not item.strip() for item in attempts)):
        raise ValueError("tolerance evidence requires at least two nonempty attempt descriptions")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("tolerance evidence requires a nonempty reason")
    return {"source": source.strip(),
            "attempts": [item.strip() for item in attempts],
            "reason": reason.strip()}


def load_tolerance_evidence(path: Path | None) -> dict[str, Any] | None:
    """Load the concise caller-provided record needed for 10–20% acceptance."""
    if path is None:
        return None
    try:
        value = json.loads(path.read_text())
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid tolerance evidence JSON: {error}") from error
    return _normalize_tolerance_evidence(value)


def _policy_unassessed(evidence: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "policy_id": PRECISION_POLICY_ID,
        "target_relative_error_limits": {
            "completion": TARGET_ERROR_LIMIT,
            "hbm_busy": TARGET_ERROR_LIMIT,
            "shared_hbm_busy": TARGET_ERROR_LIMIT,
        },
        "fallback_relative_error_limits": {
            "completion": FALLBACK_ERROR_LIMIT,
            "hbm_busy": FALLBACK_ERROR_LIMIT,
            "shared_hbm_busy": FALLBACK_ERROR_LIMIT,
        },
        "target_met": None,
        "target_unmet": None,
        "within_fallback_range": None,
        "fallback_evidence_status": "provided" if evidence else "missing",
        "fallback_evidence": evidence,
        "fallback_accepted": False,
        "serialized_runtime_work_status": "not_assessed",
        "status": "not_assessed",
    }


def _finite_number(value: Any) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value))
    except (OverflowError, ValueError):
        return False


def identify_service_framework(raw: dict[str, Any]) -> dict[str, Any]:
    """Read framework provenance and description limits from the real Plan."""
    plan = raw.get("compiled_plan", {})
    if not isinstance(plan, dict):
        return {"status": "invalid_or_unavailable", "source": None,
                "actual_plan_epoch_count": None,
                "actual_plan_description_count": None,
                "observed_frameworks": [],
                "issues": ["compiled_plan is not an object"]}
    provenance = plan.get("service_model_provenance", {})
    if not isinstance(provenance, dict):
        provenance = {}
    summary = plan.get("summary", {})
    if not isinstance(summary, dict):
        summary = {}
    descriptors = provenance.get("service_curves")
    if descriptors is None:
        descriptors = summary.get("service_curves")
    issues: list[str] = []
    if not isinstance(descriptors, list) or not descriptors:
        descriptors = []
        issues.append("compiled Plan has no service_curves provenance")

    observed = []
    for index, descriptor in enumerate(descriptors):
        if not isinstance(descriptor, dict):
            issues.append(f"service_curves[{index}] is not an object")
            continue
        model = descriptor.get("model")
        epoch_count = descriptor.get("epoch_count")
        descriptions = descriptor.get("description_count")
        budget = descriptor.get("description_budget")
        if not isinstance(model, str) or not model:
            issues.append(f"service_curves[{index}] has no model provenance")
            continue
        if not isinstance(epoch_count, int) or isinstance(epoch_count, bool) or epoch_count <= 0:
            issues.append(f"service_curves[{index}] has an invalid epoch_count")
        if not isinstance(descriptions, int) or isinstance(descriptions, bool) or descriptions <= 0:
            issues.append(f"service_curves[{index}] has an invalid description_count")
        if not isinstance(budget, int) or isinstance(budget, bool) or budget <= 0:
            issues.append(f"service_curves[{index}] has an invalid description_budget")
        observed.append({
            "model": model,
            "epoch_count": epoch_count,
            "description_count": descriptions,
            "description_budget": budget,
            "input_unit": descriptor.get("input_unit"),
            "output_unit": descriptor.get("output_unit"),
            "source_epoch_begin": descriptor.get("epoch_begin"),
            "status": ("within_declared_budget"
                       if isinstance(descriptions, int) and isinstance(budget, int)
                       and descriptions <= budget else "over_declared_budget_or_unknown"),
        })

    actual_epochs = plan.get("E")
    actual_descriptions = plan.get("description_count", provenance.get("description_count"))
    if not isinstance(actual_epochs, int) or isinstance(actual_epochs, bool) or actual_epochs <= 0:
        issues.append("compiled Plan has no valid actual epoch count")
    if (not isinstance(actual_descriptions, int) or isinstance(actual_descriptions, bool)
            or actual_descriptions <= 0):
        issues.append("compiled Plan has no valid actual description_count")
    provenance_count = provenance.get("description_count")
    plan_count = plan.get("description_count")
    if (isinstance(provenance_count, int) and isinstance(plan_count, int)
            and provenance_count != plan_count):
        issues.append("Plan and provenance description_count disagree")
    for item in observed:
        if item["status"] != "within_declared_budget":
            issues.append(f"observed service model {item['model']} exceeds or lacks its declared budget")

    return {
        "status": "identified" if observed and not issues else "invalid_or_unavailable",
        "source": provenance.get("source", "compiled_plan.summary.service_curves"),
        "actual_plan_epoch_count": actual_epochs,
        "actual_plan_description_count": actual_descriptions,
        "observed_frameworks": observed,
        "issues": issues,
    }


def _validate_curve_serialization(raw: dict[str, Any]) -> dict[str, int]:
    """Reconstruct descriptions from each epoch's independent Activity segments."""
    plan = raw["compiled_plan"]
    curve_activities = point_count = internal_breakpoints = 0
    epoch_description_counts: list[int] = []
    epochs = plan.get("epochs", [])
    if not isinstance(epochs, list):
        raise UnsupportedScenario("compiled Plan epochs must be an array")
    for epoch in epochs:
        if not isinstance(epoch, dict) or not isinstance(epoch.get("activities", []), list):
            raise UnsupportedScenario("compiled Plan epoch activities must be an array")
        activity_description_count = 0
        for activity in epoch.get("activities", []):
            if not isinstance(activity, dict):
                raise UnsupportedScenario("compiled Plan activity must be an object")
            if "credit_curve" not in activity:
                raise UnsupportedScenario("driver omitted Activity.credit_curve metadata")
            curve = activity["credit_curve"]
            if not isinstance(curve, list):
                raise UnsupportedScenario("Activity.credit_curve must be an array")
            if not curve:
                activity_description_count += 1
                continue
            activity_description_count += max(1, len(curve) - 1)
            curve_activities += 1
            previous_progress = -1.0
            previous_credit: list[float] | None = None
            for point in curve:
                if not isinstance(point, dict):
                    raise UnsupportedScenario("Activity.credit_curve points must be objects")
                progress = point.get("progress")
                cumulative = point.get("cumulative")
                if (not _finite_number(progress) or not 0 <= float(progress) <= 1
                        or not isinstance(cumulative, list) or len(cumulative) != 2
                        or not all(_finite_number(value) for value in cumulative)):
                    raise UnsupportedScenario("Activity.credit_curve contains a non-finite or malformed point")
                progress = float(progress)
                cumulative = [float(value) for value in cumulative]
                if progress <= previous_progress:
                    raise UnsupportedScenario("Activity.credit_curve progress must be strictly increasing")
                if previous_credit is None and (progress != 0.0 or cumulative != [0.0, 0.0]):
                    raise UnsupportedScenario("Activity.credit_curve must begin at zero progress and credit")
                previous_progress = progress
                previous_credit = cumulative
                point_count += 1
            if curve[-1]["progress"] != 1.0:
                raise UnsupportedScenario("Activity.credit_curve must end at progress one")
            internal_breakpoints += max(0, len(curve) - 2)
            if previous_credit is not None:
                delta = activity.get("credit_delta")
                if (not isinstance(delta, list) or len(delta) != 2
                        or not all(_finite_number(value) for value in delta)
                        or any(abs(previous_credit[i] - float(delta[i])) >
                               1e-10 * max(1.0, abs(previous_credit[i]), abs(float(delta[i])))
                               for i in range(2))):
                    raise UnsupportedScenario("Activity.credit_curve endpoint disagrees with credit_delta")
        epoch_description_counts.append(max(1, activity_description_count))
    reconstructed_descriptions = sum(epoch_description_counts)
    actual_epoch_count = plan.get("E")
    actual_description_count = plan.get("description_count")
    if actual_epoch_count != len(epochs):
        raise UnsupportedScenario("compiled Plan E disagrees with serialized epochs")
    if actual_description_count != reconstructed_descriptions:
        raise UnsupportedScenario(
            "compiled Plan description_count disagrees with serialized epochs and curve breakpoints")
    return {"actual_plan_epoch_count": len(epochs),
            "activities_with_piecewise_curve": curve_activities,
            "serialized_curve_points": point_count,
            "serialized_internal_curve_breakpoints": internal_breakpoints,
            "reconstructed_epoch_description_counts": epoch_description_counts,
            "reconstructed_description_count": reconstructed_descriptions}


def _validate_framework_ranges(raw: dict[str, Any]) -> None:
    """Check declared service descriptions against actual activity segments."""
    plan = raw["compiled_plan"]
    provenance = plan.get("service_model_provenance", {})
    descriptors = provenance.get("service_curves")
    if descriptors is None:
        descriptors = plan.get("summary", {}).get("service_curves", [])
    epochs = plan.get("epochs", [])
    for descriptor in descriptors:
        begin = descriptor.get("epoch_begin")
        count = descriptor.get("epoch_count")
        if (not isinstance(begin, int) or isinstance(begin, bool) or begin < 0
                or not isinstance(count, int) or isinstance(count, bool) or count <= 0
                or begin + count > len(epochs)):
            raise UnsupportedScenario("service model epoch range is outside the actual Plan")
        reconstructed = 0
        for epoch in epochs[begin:begin + count]:
            activity_descriptions = 0
            for activity in epoch.get("activities", []):
                curve = activity.get("credit_curve", [])
                activity_descriptions += max(1, len(curve) - 1) if curve else 1
            reconstructed += max(1, activity_descriptions)
        if descriptor.get("description_count") != reconstructed:
            raise UnsupportedScenario(
                "service model description_count disagrees with its serialized activity segments and epoch range")


def _validate_coverage(raw: dict[str, Any]) -> None:
    if raw.get("measurement_source") != "compiled_plan_and_actual_cpp_runtime":
        raise UnsupportedScenario("oracle requires actual compiled Plan and C++ Runtime observations")
    hardware = raw.get("synthetic_hardware", {})
    plan = raw.get("compiled_plan", {})
    if not isinstance(hardware, dict) or not isinstance(plan, dict):
        raise UnsupportedScenario("hardware and compiled Plan provenance must be objects")
    task = plan.get("task", {})
    shape = task.get("tensor_shapes", {})
    if not isinstance(task, dict) or not isinstance(shape, dict):
        raise UnsupportedScenario("compiled GEMM task shape metadata is malformed")
    summary = plan.get("summary", {})
    if not isinstance(summary, dict):
        raise UnsupportedScenario("compiled Plan summary must be an object")
    if (hardware.get("core_count") != 1 or hardware.get("core_columns_nx") != 1
            or hardware.get("core_rows_ny") != 1):
        raise UnsupportedScenario("discrete oracle supports only a complete single-Core 1x1 grid")
    if summary.get("dataflow_id") != "output_stationary_v2":
        raise UnsupportedScenario("discrete oracle supports only an actual output-stationary Plan")
    if task.get("execution_profile") != "fp32_vec_matrix_assumed_v1":
        raise UnsupportedScenario("discrete oracle supports only fp32_vec_matrix_assumed_v1")
    storage_dtype = task.get("storage_dtype", {})
    if not isinstance(storage_dtype, dict):
        raise UnsupportedScenario("compiled Plan storage_dtype metadata is malformed")
    if any(storage_dtype.get(name) != "fp32"
           for name in ("weights", "activations", "output")):
        raise UnsupportedScenario("discrete oracle supports only FP32 storage")
    if any(not isinstance(shape.get(name), int) or shape[name] <= 0
           for name in ("m", "n", "k")):
        raise UnsupportedScenario("actual Plan is missing a positive GEMM shape")

    capacities = hardware.get("resource_capacities")
    if not isinstance(capacities, list) or len(capacities) != len(reference.RESOURCE_NAMES):
        raise UnsupportedScenario("actual Runtime report does not contain all 13 resource capacities")
    for descriptor in capacities:
        if (not isinstance(descriptor, dict)
                or not _finite_number(descriptor.get("capacity"))
                or float(descriptor["capacity"]) < 0):
            raise UnsupportedScenario("non-finite or negative resource capacity is outside oracle coverage")
    for name in ("COMPUTE_GATE", "HBM0", "HBM1"):
        capacity = float(capacities[reference.RESOURCE_IDS[name]]["capacity"])
        if capacity <= 0:
            raise UnsupportedScenario(f"oracle requires a positive {name} capacity")


def _make_discrete_waves(raw: dict[str, Any]):
    """Use the existing independent wave formulas, bypassing its old E guard.

    Framework provenance is validated independently against the actual Plan.
    Reuse the independent geometry/count formulas with their historical E5
    guard explicitly disabled, without changing any measured Plan metadata.
    """
    return reference.make_waves(raw, require_five_epoch_fixture=False)


def _absolute_errors(comparison: dict[str, Any], actual: dict[str, Any],
                     oracle: dict[str, Any], concurrent: bool) -> dict[str, Any]:
    if concurrent:
        actual_finishes = actual["finish_s_by_request"]
        expected_finishes = oracle["finish_s_by_request"]
        finish_errors = {
            name: abs(float(actual_finishes[name]) - float(expected_finishes[name]))
            for name in ("comp0", "comp1")
        }
    else:
        actual_finishes = {"solo": float(actual["finish_s"])}
        expected_finishes = {"solo": float(oracle["finish_s_by_request"]["comp0"])}
        finish_errors = {"solo": abs(actual_finishes["solo"] - expected_finishes["solo"])}
    actual_counters = actual["counters"]
    actual_hbm_busy = float(actual_counters["hbm_busy_ns"]) / reference.NS_PER_SECOND
    actual_shared = float(actual_counters["hbm_shared_ns"]) / reference.NS_PER_SECOND
    return {
        "finish_time_s_by_request": finish_errors,
        "hbm_busy_s": abs(actual_hbm_busy - float(oracle["hbm_busy_s"])),
        "shared_hbm_busy_s": abs(actual_shared - float(oracle["hbm_shared_s"])),
        "reference_shared_hbm_duration_s": float(oracle["hbm_shared_s"]),
        "actual_shared_hbm_duration_s": actual_shared,
        "relative_checks_status": comparison["status"],
    }


def _compare(actual: dict[str, Any], oracle: dict[str, Any], phase: float,
             concurrent: bool) -> dict[str, Any]:
    result = reference.compare_measurement(
        actual, oracle, phase, concurrent,
        completion_limit=TARGET_ERROR_LIMIT,
        busy_limit=TARGET_ERROR_LIMIT)
    result["absolute_errors_and_shared_reference"] = _absolute_errors(
        result, actual, oracle, concurrent)
    errors = result["relative_errors"]
    completion_error = max(errors["finish_by_request"].values(), default=math.inf)
    shared_required = result["limits"]["shared_hbm_busy"] is not None
    hbm_busy_error = errors["hbm_busy"]
    shared_busy_error = errors["shared_hbm_busy"]
    target_checks = {
        "completion": completion_error <= TARGET_ERROR_LIMIT,
        "hbm_busy": hbm_busy_error <= TARGET_ERROR_LIMIT,
        "shared_hbm_busy": (shared_busy_error <= TARGET_ERROR_LIMIT
                             if shared_required else True),
    }
    fallback_checks = {
        "completion": completion_error <= FALLBACK_ERROR_LIMIT,
        "hbm_busy": hbm_busy_error <= FALLBACK_ERROR_LIMIT,
        "shared_hbm_busy": (shared_busy_error <= FALLBACK_ERROR_LIMIT
                             if shared_required else True),
    }
    target_met = all(target_checks.values())
    within_fallback = all(fallback_checks.values())
    result["precision_assessment"] = {
        "status": ("target_met" if target_met else
                   "within_fallback_range" if within_fallback else
                   "outside_fallback_range"),
        "target_met": target_met,
        "within_fallback_range": within_fallback,
        "relative_errors": {
            "completion": completion_error,
            "hbm_busy": hbm_busy_error,
            "shared_hbm_busy": shared_busy_error,
        },
        "target_checks": target_checks,
        "fallback_checks": fallback_checks,
        "target_limit": TARGET_ERROR_LIMIT,
        "fallback_limit": FALLBACK_ERROR_LIMIT,
        "fallback_accepted": False,
    }
    return result


def _serialized_runtime_work_reconciliation(
        comparisons: list[tuple[str, dict[str, Any]]]) -> dict[str, Any]:
    """Summarize the exact served-work diagnostic without hiding its absence."""
    failed_statuses = {
        "failed", "outside_relative_tolerance", "incomplete_history",
    }
    scenarios = []
    for name, comparison in comparisons:
        reconciliation = comparison.get(
            "runtime_cumulative_serialization_reconciliation")
        if not isinstance(reconciliation, dict):
            raw_status = "missing"
            available = None
        else:
            raw_status = reconciliation.get("status")
            available = reconciliation.get("available")
            if not isinstance(raw_status, str) or not raw_status:
                raw_status = "missing"
        if raw_status in failed_statuses:
            outcome = "failed"
        elif raw_status == "within_relative_tolerance" and available is True:
            outcome = "verified"
        elif raw_status == "not_available" and available is False:
            outcome = "not_available"
        elif raw_status == "missing":
            outcome = "missing"
        else:
            # Unknown or internally inconsistent states must not silently count
            # as an exact-work pass.
            outcome = "unrecognized_status"
        scenarios.append({
            "scenario": name,
            "status": raw_status,
            "outcome": outcome,
            "available": available,
            "verified": outcome == "verified",
            "reason": (reconciliation.get("reason")
                       if isinstance(reconciliation, dict) else None),
            "missing_counter_fields": (
                reconciliation.get("missing_counter_fields", [])
                if isinstance(reconciliation, dict) else []),
        })

    outcomes = {item["outcome"] for item in scenarios}
    if outcomes & {"failed", "unrecognized_status"}:
        status = "failed"
    elif "missing" in outcomes:
        status = "missing"
    elif "not_available" in outcomes:
        status = "not_available"
    else:
        status = "within_relative_tolerance"
    blocks_acceptance = bool(outcomes & {
        "failed", "missing", "unrecognized_status",
    })
    return {
        "status": status,
        "all_available_scenarios_within_tolerance": bool(scenarios) and all(
            item["verified"] for item in scenarios),
        "blocks_acceptance": blocks_acceptance,
        "scenarios": scenarios,
        "interpretation": (
            "Serialized Runtime work counters reconcile within the declared tolerance for every scenario."
            if status == "within_relative_tolerance" else
            "Serialized Runtime work counters are unavailable; precision results do not verify exact served work."
            if status == "not_available" else
            "Serialized Runtime work reconciliation failed or its result is missing or unrecognized."),
    }


def build_report(raw: dict[str, Any], *, entry_parameters: dict[str, Any] | None = None,
                 tolerance_evidence: dict[str, Any] | None = None) -> dict[str, Any]:
    tolerance_evidence = _normalize_tolerance_evidence(tolerance_evidence)
    framework = identify_service_framework(raw)
    if framework["status"] != "identified":
        raise UnsupportedScenario("service framework metadata is incomplete or inconsistent: "
                                  + "; ".join(framework["issues"]))
    _validate_coverage(raw)
    curve_coverage = _validate_curve_serialization(raw)
    _validate_framework_ranges(raw)
    waves, expected = _make_discrete_waves(raw)
    ledger = reference.check_ledger(raw, expected)
    capacities = reference.make_capacities(raw)

    solo_oracle = grouped.discrete_grouped(
        waves, expected["true_output_tiles"], [0.0], capacities)
    solo = _compare(raw["standalone"], solo_oracle, 0.0, False)
    concurrent = []
    actual_concurrent = raw.get("two_comp", [])
    if not isinstance(actual_concurrent, list):
        raise UnsupportedScenario("actual Runtime two_comp observations must be an array")
    for actual in actual_concurrent:
        offset = float(actual["phase_offset_s"])
        oracle = grouped.discrete_grouped(
            waves, expected["true_output_tiles"], [0.0, offset], capacities)
        concurrent.append(_compare(actual, oracle, offset, True))

    scenario_coverage = {
        "standalone_scenario_observed": isinstance(raw.get("standalone"), dict),
        "actual_concurrent_scenarios_observed": len(actual_concurrent),
        "concurrent_completion_and_shared_busy_compared": bool(concurrent),
    }
    comparisons = [solo, *concurrent]
    target_met = all(item["precision_assessment"]["target_met"]
                     for item in comparisons)
    within_fallback = all(item["precision_assessment"]["within_fallback_range"]
                          for item in comparisons)
    runtime_work_reconciliation = _serialized_runtime_work_reconciliation([
        ("standalone", solo),
        *((f"two_comp[{index}]", item)
          for index, item in enumerate(concurrent)),
    ])
    scenario_complete = (scenario_coverage["standalone_scenario_observed"]
                         and bool(concurrent))
    work_and_scenario_valid = scenario_complete and ledger["status"] == "passed"
    exact_work_allows_acceptance = not runtime_work_reconciliation["blocks_acceptance"]
    passed = (work_and_scenario_valid and exact_work_allows_acceptance
              and target_met)
    fallback_accepted = (work_and_scenario_valid and not target_met
                        and exact_work_allows_acceptance and within_fallback
                        and tolerance_evidence is not None)
    status = ("passed" if passed else
              "accepted_with_fallback" if fallback_accepted else "failed")
    for item in comparisons:
        assessment = item["precision_assessment"]
        assessment["fallback_accepted"] = (
            fallback_accepted and not assessment["target_met"]
            and assessment["within_fallback_range"])
        if assessment["fallback_accepted"]:
            assessment["status"] = "accepted_with_fallback"
        elif not assessment["target_met"] and assessment["within_fallback_range"]:
            assessment["status"] = ("fallback_evidence_missing" if tolerance_evidence is None
                                     else "fallback_blocked_by_other_checks")
    plan = raw["compiled_plan"]
    summary = plan["summary"]
    available = [{
        "model": item["model"],
        "observed_epoch_count": item["epoch_count"],
        "description_count": item["description_count"],
        "description_budget": item["description_budget"],
        "status": item["status"],
    } for item in framework["observed_frameworks"]]
    return {
        "schema_version": 2,
        "oracle": "grouped_gemm_oracle.independent_discrete_k_consumption_waves",
        "status": status,
        "status_scope": "one_actual_compiled_plan_and_cpp_runtime_fixture",
        "acceptance_scope": (
            "precision comparison plus compiled-plan work ledger; when serialized Runtime "
            "work is not_available, acceptance is precision-only and does not validate "
            "exact served-work counters"),
        "fixture_kind": raw["synthetic_hardware"].get("fixture_kind"),
        "measurement_source": raw.get("measurement_source"),
        "entry_parameters": entry_parameters or {},
        "precision_policy": {
            "policy_id": PRECISION_POLICY_ID,
            "target_relative_error_limits": {
                "completion": TARGET_ERROR_LIMIT,
                "hbm_busy": TARGET_ERROR_LIMIT,
                "shared_hbm_busy": TARGET_ERROR_LIMIT,
            },
            "fallback_relative_error_limits": {
                "completion": FALLBACK_ERROR_LIMIT,
                "hbm_busy": FALLBACK_ERROR_LIMIT,
                "shared_hbm_busy": FALLBACK_ERROR_LIMIT,
            },
            "target_met": target_met,
            "target_unmet": not target_met,
            "within_fallback_range": within_fallback,
            "fallback_evidence_status": "provided" if tolerance_evidence else "missing",
            "fallback_evidence": tolerance_evidence,
            "fallback_accepted": fallback_accepted,
            "serialized_runtime_work_status": runtime_work_reconciliation["status"],
            "status": ("target_met" if passed else
                       "accepted_with_fallback" if fallback_accepted else
                       "runtime_serialized_work_reconciliation_failed"
                       if runtime_work_reconciliation["blocks_acceptance"] else
                       "work_or_scenario_checks_failed"
                       if not work_and_scenario_valid else
                       "fallback_not_accepted" if within_fallback and not target_met else
                       "outside_fallback_range" if not within_fallback else
                       "work_or_scenario_checks_failed"),
        },
        "compiled_plan": {
            "fingerprint": plan.get("fingerprint"),
            "compiler_path": plan.get("compiler_path"),
            "E": plan.get("E"),
            "description_count": plan.get("description_count"),
            "candidate_count": summary.get("candidate_count"),
            "selected_template": summary.get("selected_template", summary.get("template_id")),
            "dataflow_id": summary.get("dataflow_id"),
            "tiles": {key: summary.get(key) for key in ("tm", "tn", "tk")},
            "service_model_provenance": framework,
            "credit_curve_coverage": curve_coverage,
            "mapping_selection": raw.get("mapping_selection"),
            "summary": summary,
        },
        "fixture_coverage": {
            "shape": expected["shape"],
            "core_count": raw["synthetic_hardware"]["core_count"],
            "dataflow": summary["dataflow_id"],
            "profile": "fp32_vec_matrix_assumed_v1; FP32 weights/activations/output",
            "reference": "existing grouped owner Q/alpha discrete wave oracle",
            "reference_geometry_adapter": (
                "shared discrete geometry helper with legacy E5 guard disabled explicitly; "
                "framework metadata validated independently from the actual Plan"),
            "hardware_scope": "synthetic model-to-model capacities; no silicon evidence",
            "coverage_limits": [
                "single complete 1x1 Core only",
                "one GEMM shape and selected mapping per report",
                "FP32 storage and fp32_vec_matrix_assumed_v1 only",
                "other hardware domains use explicit finite nonbinding fixture capacities",
            ],
        },
        "independent_expected_work": expected,
        "compiled_plan_ledger": ledger,
        "runtime_cumulative_serialization_reconciliation": runtime_work_reconciliation,
        "comparisons": {"standalone": solo, "two_comp": concurrent},
        "scenario_coverage": scenario_coverage,
        "acceptance_limits": {
            "standalone_finish_relative_error": TARGET_ERROR_LIMIT,
            "concurrent_finish_relative_error": TARGET_ERROR_LIMIT,
            "hbm_busy_relative_error": TARGET_ERROR_LIMIT,
            "shared_hbm_busy_relative_error": TARGET_ERROR_LIMIT,
        },
        "available_granularity_levels": {
            "source": "actual compiled Plan service_model_provenance.service_curves",
            "observed": available,
            "minimum_certified_framework": None,
            "status": "fixture_comparison_only",
            "comparison_status": status,
        },
        "status_reason": (
            "all measured scenarios meet the 10% target, the compiled-plan ledger passed, and serialized Runtime work is within tolerance"
            if passed and runtime_work_reconciliation[
                "all_available_scenarios_within_tolerance"] else
            "precision-only comparison meets the 10% target and the compiled-plan ledger passed; exact serialized Runtime work counters are unavailable"
            if passed else
            "serialized Runtime work reconciliation failed or is missing; precision tolerance cannot waive it"
            if runtime_work_reconciliation["blocks_acceptance"] else
            "work reconciliation or required scenario coverage failed; precision tolerance cannot waive it"
            if not work_and_scenario_valid else
            "20% precision fallback accepted with caller-provided attempts and a reason; exact serialized Runtime work counters are unavailable"
            if fallback_accepted and runtime_work_reconciliation["status"] == "not_available" else
            "20% fallback accepted with caller-provided attempts and a reason, and serialized Runtime work was verified"
            if fallback_accepted else
            "20% fallback range requires documented attempts and a reason"
            if within_fallback and not target_met else
            "work reconciliation, scenario coverage, or at least one error exceeded the 20% range"),
        "notes": [
            "Only actual driver Runtime scenarios are compared; the Python side enumerates the small reference waves.",
            "No new E-level certification or minimum-granularity claim is made.",
            "The 10% target and 20% fallback are an offline comparison policy; fallback evidence is caller-provided and not independently judged by this tool.",
            "Runtime serialized work reconciliation is reported separately; not_available means exact served-work counts are unverified, not passed.",
            "A failed precision comparison is a valid tool result and returns CLI status 1.",
        ],
    }


def _json_text(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"


def _write_json(path: Path | None, value: Any) -> None:
    content = _json_text(value)
    if path is None:
        print(content, end="")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)


def _driver_arguments(args: argparse.Namespace) -> list[str]:
    shape = (args.m, args.n, args.k)
    if any(value is not None for value in shape) and not all(value is not None for value in shape):
        raise ValueError("--m, --n, and --k must be supplied together")
    if any(value is not None and value <= 0 for value in (*shape, args.tm, args.tn, args.tk)):
        raise ValueError("shape and tile values must be positive integers")
    values = []
    for name in ("m", "n", "k", "tm", "tn", "tk"):
        value = getattr(args, name)
        if value is not None:
            values.extend((f"--{name}", str(value)))
    return values


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--driver", type=Path,
                        default=Path("work/build-die/coarse_compiled_gemm_oracle_driver"))
    parser.add_argument("--raw", type=Path, help="load a previous actual driver JSON")
    parser.add_argument("--raw-output", type=Path,
                        help="save actual driver JSON (defaults beside --output when running the driver)")
    parser.add_argument("--output", type=Path, help="save report JSON; stdout when omitted")
    parser.add_argument("--tolerance-evidence", type=Path,
                        help="JSON source/attempts/reason record required to accept 10–20% error")
    parser.add_argument("--m", type=int)
    parser.add_argument("--n", type=int)
    parser.add_argument("--k", type=int)
    parser.add_argument("--tm", type=int)
    parser.add_argument("--tn", type=int)
    parser.add_argument("--tk", type=int)
    args = parser.parse_args(argv)

    parameters = {
        "driver": str(args.driver),
        "raw_input": str(args.raw) if args.raw else None,
        "raw_output": str(args.raw_output) if args.raw_output else None,
        "report_output": str(args.output) if args.output else None,
        "tolerance_evidence": (str(args.tolerance_evidence)
                               if args.tolerance_evidence else None),
        "requested_shape": {key: getattr(args, key) for key in ("m", "n", "k")},
        "requested_tiles": {key: getattr(args, key) for key in ("tm", "tn", "tk")},
    }
    raw: dict[str, Any] | None = None
    raw_output = args.raw_output
    tolerance_evidence = None
    try:
        driver_args = _driver_arguments(args)
        parameters["driver_arguments"] = driver_args
        if (args.raw_output is not None and args.output is not None
                and args.raw_output.resolve() == args.output.resolve()):
            raise ValueError("--raw-output and --output must use different paths")
        tolerance_evidence = load_tolerance_evidence(args.tolerance_evidence)
        if args.raw:
            if driver_args:
                raise ValueError("shape/tile selection belongs to driver runs; omit it with --raw")
            raw = json.loads(args.raw.read_text())
        else:
            if not args.driver.is_file():
                raise UnsupportedScenario(f"actual C++ driver is unavailable: {args.driver}")
            completed = subprocess.run([str(args.driver.resolve()), *driver_args],
                                       check=False, capture_output=True, text=True)
            if completed.returncode != 0:
                reason = completed.stderr.strip() or f"driver exited {completed.returncode}"
                raise UnsupportedScenario(f"actual C++ driver rejected or could not run scenario: {reason}")
            raw = json.loads(completed.stdout)
            if raw_output is None:
                raw_output = (args.output.with_name(args.output.stem + ".raw.json")
                              if args.output is not None
                              else Path("work/analytical_service_oracle.raw.json"))
        if raw_output is not None:
            _write_json(raw_output, raw)
        parameters["raw_output"] = str(raw_output) if raw_output else None
        if raw is not None and args.raw:
            parameters["raw_driver_source"] = str(args.raw)
        report = build_report(raw, entry_parameters=parameters,
                              tolerance_evidence=tolerance_evidence)
    except UnsupportedScenario as error:
        report = {
            "schema_version": 2,
            "oracle": "grouped_gemm_oracle.independent_discrete_k_consumption_waves",
            "status": "unsupported",
            "status_scope": "no precision comparison performed",
            "reason": str(error),
            "entry_parameters": parameters,
            "precision_policy": _policy_unassessed(tolerance_evidence),
            "available_granularity_levels": {"observed": [], "minimum_certified_framework": None},
        }
        _write_json(args.output, report)
        return 2
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        report = {
            "schema_version": 2,
            "oracle": "grouped_gemm_oracle.independent_discrete_k_consumption_waves",
            "status": "not_run",
            "status_scope": "invalid input or tool execution error",
            "reason": str(error),
            "entry_parameters": parameters,
            "precision_policy": _policy_unassessed(tolerance_evidence),
        }
        _write_json(args.output, report)
        return 2

    _write_json(args.output, report)
    return 0 if report["status"] in {"passed", "accepted_with_fallback"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
