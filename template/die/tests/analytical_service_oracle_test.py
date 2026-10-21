#!/usr/bin/env python3
"""Focused state/provenance tests for the offline analytical service oracle."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "validation"))

import analytical_service_oracle as oracle
import compiled_gemm_oracle as reference


def make_fixture(relative_error: float = 0.0) -> dict:
    capacities = []
    for resource, name in enumerate(reference.RESOURCE_NAMES):
        capacity = (1.0 if name == "COMPUTE_GATE" else
                    50_000_000.0 if name in ("HBM0", "HBM1") else 1.0e30)
        capacities.append({"resource_id": resource, "id": name,
                           "enabled": True, "capacity": capacity})
    summary = {
        "m": 1, "n": 1, "k": 1,
        "tm": 1, "tn": 1, "tk": 1,
        "dataflow_id": "output_stationary_v2",
        "output_round_count": 1,
        "consumption_wave_count": 1,
        "hbm_read_bytes_total": 0,
        "hbm_write_bytes_total": 0,
        "vec_equivalent_ops": 0,
        "conversion_elements": 0,
        "int4_input_bytes": 0,
        "executed_int4_ops": 0,
        "sram_read_bytes": 0,
        "sram_write_bytes": 0,
        "local_bus_bytes": 0,
        "logical_tensor_bytes": 0,
        "sram_peak_bytes_per_core": 0,
        "source_matrix_ops": 0,
        "executed_matrix_ops": 0,
        "service_curves": [{
            "model": "analytical_cumulative_os_v3",
            "epoch_begin": 0,
            "epoch_count": 4,
            "description_count": 7,
            "description_budget": 8,
            "input_unit": "one_output_tile_K_consumption",
            "output_unit": "one_output_tile",
        }],
    }
    raw = {
        "schema_version": 1,
        "measurement_source": "compiled_plan_and_actual_cpp_runtime",
        "synthetic_hardware": {
            "fixture_kind": "synthetic_model_to_model",
            "core_count": 1,
            "core_columns_nx": 1,
            "core_rows_ny": 1,
            "core_sram_bytes_per_core": 1_000_000,
            "p_vec_per_core_ops_per_s": 1.0e9,
            "rho_sram": 0.9,
            "eta_vec": 0.02,
            "eta_hbm_matrix": 1.0,
            "resource_capacities": capacities,
        },
        "compiled_plan": {
            "compiler_path": "actual Mapping enumeration",
            "fingerprint": "fixture-plan",
            "E": 4,
            "description_count": 7,
            "service_model_provenance": {
                "source": "compiled Plan summary.service_curves and Activity.credit_curve",
                "service_curves": summary["service_curves"],
                "plan_epoch_count": 4,
                "description_count": 7,
            },
            "task": {
                "op_kind": "gemm",
                "tensor_shapes": {"m": 1, "n": 1, "k": 1},
                "storage_dtype": {"weights": "fp32", "activations": "fp32",
                                  "output": "fp32"},
                "execution_profile": "fp32_vec_matrix_assumed_v1",
            },
            "summary": summary,
            "epochs": [],
        },
        "standalone": {
            "finish_s": 1.0,
            "counters": {
                "hbm_busy_ns": 0,
                "hbm_shared_ns": 0,
                "compute_busy_ns": 0,
                "internal_events": 0,
                "allocation_updates": 0,
                "row_busy_ns": [0, 0],
                "row_shared_ns": [0, 0],
                "accounting_history_complete": True,
            },
        },
        "two_comp": [],
    }
    waves, expected = oracle._make_discrete_waves(raw)
    summary.update({
        "hbm_read_bytes_total": expected["hbm_read_bytes"],
        "hbm_write_bytes_total": expected["hbm_write_bytes"],
        "vec_equivalent_ops": expected["vec_equivalent_ops"],
        "sram_read_bytes": expected["sram_read_bytes"],
        "sram_write_bytes": expected["sram_write_bytes"],
        "local_bus_bytes": expected["local_bus_bytes"],
        "logical_tensor_bytes": expected["logical_tensor_bytes"],
        "sram_peak_bytes_per_core": expected["sram_peak_bytes_per_core"],
        "source_matrix_ops": expected["vec_equivalent_ops"],
        "executed_matrix_ops": expected["vec_equivalent_ops"],
    })
    work = [0.0] * len(reference.RESOURCE_NAMES)
    work[reference.RESOURCE_IDS["COMPUTE_GATE"]] = expected["compute_work_s"]
    work[reference.RESOURCE_IDS["SRAM_READ"]] = expected["sram_read_bytes"]
    work[reference.RESOURCE_IDS["SRAM_WRITE"]] = expected["sram_write_bytes"]
    work[reference.RESOURCE_IDS["LOCAL"]] = expected["local_bus_bytes"]
    for row, name in enumerate(("HBM0", "HBM1")):
        work[reference.RESOURCE_IDS[name]] = (
            expected["hbm_read_by_row"][row] + expected["hbm_write_by_row"][row])
    activity = {
        "id": 1,
        "role": "compute",
        "work_by_resource": work,
        "hbm_bytes_by_row_direction": [
            [expected["hbm_read_by_row"][0], expected["hbm_write_by_row"][0]],
            [expected["hbm_read_by_row"][1], expected["hbm_write_by_row"][1]],
        ],
        "int4_ops": 0,
        "vec_eqops": expected["vec_equivalent_ops"],
        "canonical_q": 1,
        "private_time_s": 0,
        "credit_delta": [-1, 0.9],
        "credit_curve": [
            {"progress": 0.0, "cumulative": [0.0, 0.0]},
            {"progress": 0.5, "cumulative": [-0.5, 0.0]},
            {"progress": 1.0, "cumulative": [-1.0, 0.9]},
        ],
    }
    next_aux_activity_id = 100

    def zero_activity(role: str, curve: list[dict] | None = None) -> dict:
        nonlocal next_aux_activity_id
        item = {
            "id": next_aux_activity_id,
            "role": role,
            "work_by_resource": [0.0] * len(reference.RESOURCE_NAMES),
            "hbm_bytes_by_row_direction": [[0, 0], [0, 0]],
            "int4_ops": 0,
            "vec_eqops": 0,
            "canonical_q": 0,
            "private_time_s": 0.0,
            "credit_delta": [0.0, 0.0],
            "credit_curve": curve or [],
        }
        next_aux_activity_id += 1
        return item

    two_segment_zero_curve = [
        {"progress": 0.0, "cumulative": [0.0, 0.0]},
        {"progress": 0.5, "cumulative": [0.0, 0.0]},
        {"progress": 1.0, "cumulative": [0.0, 0.0]},
    ]
    raw["compiled_plan"]["epochs"] = [
        {"name": f"epoch{index}", "repeat_count": 1,
         "credit_initial": [0, 0], "credit_capacity": [2.0, 2.0],
         "activities": (
             [zero_activity("load")] if index == 0 else
             [zero_activity("load", two_segment_zero_curve), activity] if index == 1 else
             [zero_activity("store")])}
        for index in range(4)
    ]
    _set_measurement_error(raw, relative_error)
    return raw


def _set_measurement_error(raw: dict, relative_error: float) -> None:
    """Populate synthetic Runtime results at a controlled error from the oracle."""
    waves, expected = oracle._make_discrete_waves(raw)
    capacities = reference.make_capacities(raw)
    solo = oracle.grouped.discrete_grouped(
        waves, expected["true_output_tiles"], [0.0], capacities)
    concurrent = oracle.grouped.discrete_grouped(
        waves, expected["true_output_tiles"], [0.0, 0.0], capacities)
    scale = 1.0 + relative_error

    def ns(seconds: float) -> int:
        return int(round(seconds * scale * reference.NS_PER_SECOND))

    def counters(result: dict) -> dict:
        return {
            "hbm_busy_ns": ns(result["hbm_busy_s"]),
            "hbm_shared_ns": ns(result["hbm_shared_s"]),
            "compute_busy_ns": ns(result["compute_busy_s"]),
            "internal_events": result["event_count"],
            "allocation_updates": 0,
            "row_busy_ns": [ns(value) for value in result["row_busy_s"]],
            "row_shared_ns": [ns(value) for value in result["row_shared_s"]],
            "accounting_history_complete": True,
        }

    raw["standalone"]["finish_s"] = (
        solo["finish_s_by_request"]["comp0"] * scale)
    raw["standalone"]["counters"] = counters(solo)
    raw["two_comp"] = [{
        "phase_offset_s": 0.0,
        "finish_s_by_request": {
            name: float(finish) * scale
            for name, finish in concurrent["finish_s_by_request"].items()
        },
        "counters": counters(concurrent),
    }]

    def serialized_work(result: dict, request_count: int) -> dict:
        ledger = result["discrete_wave_work_ledger"]
        return {
            "hbm_served_bytes_by_row_direction_exact": [
                [str(ledger["hbm_read_bytes_by_row_per_request"][row] * request_count),
                 str(ledger["hbm_write_bytes_by_row_per_request"][row] * request_count)]
                for row in (0, 1)],
            "served_by_category": {
                "fixture": {
                    "vec_eqops_exact_decimal": str(
                        ledger["vec_equivalent_ops_per_request"] * request_count),
                },
            },
        }

    raw["standalone"]["counters"].update(serialized_work(solo, 1))
    raw["two_comp"][0]["counters"].update(serialized_work(concurrent, 2))


class AnalyticalServiceOracleTest(unittest.TestCase):
    def test_identifies_actual_framework_metadata_without_epoch_certificate(self):
        raw = make_fixture()
        metadata = oracle.identify_service_framework(raw)
        self.assertEqual(metadata["status"], "identified")
        self.assertEqual(metadata["actual_plan_epoch_count"],
                         len(raw["compiled_plan"]["epochs"]))
        self.assertEqual(metadata["actual_plan_description_count"],
                         raw["compiled_plan"]["description_count"])
        self.assertEqual(metadata["observed_frameworks"][0]["model"],
                         raw["compiled_plan"]["service_model_provenance"]
                         ["service_curves"][0]["model"])
        self.assertEqual(metadata["observed_frameworks"][0]["description_count"], 7)
        self.assertEqual(metadata["observed_frameworks"][0]["description_budget"], 8)

    def test_description_count_sums_activity_segments_per_epoch(self):
        raw = make_fixture()
        coverage = oracle._validate_curve_serialization(raw)
        self.assertEqual(coverage["reconstructed_epoch_description_counts"], [1, 4, 1, 1])
        self.assertEqual(coverage["reconstructed_description_count"], 7)
        oracle._validate_framework_ranges(raw)

        empty_epochs = make_fixture()
        empty_epochs["compiled_plan"]["epochs"][0]["activities"] = []
        empty_epochs["compiled_plan"]["epochs"][2]["activities"] = []
        empty_epochs["compiled_plan"]["epochs"][3]["activities"] = []
        empty_coverage = oracle._validate_curve_serialization(empty_epochs)
        self.assertEqual(empty_coverage["reconstructed_epoch_description_counts"], [1, 4, 1, 1])

        raw["compiled_plan"]["description_count"] = 6
        with self.assertRaisesRegex(oracle.UnsupportedScenario, "description_count"):
            oracle._validate_curve_serialization(raw)

    def test_single_cohort_shape_can_report_three_actual_epochs(self):
        raw = make_fixture()
        plan = raw["compiled_plan"]
        plan["epochs"].pop()
        plan["E"] = len(plan["epochs"])
        plan["description_count"] = 6
        plan["service_model_provenance"]["plan_epoch_count"] = len(plan["epochs"])
        plan["service_model_provenance"]["description_count"] = 6
        plan["service_model_provenance"]["service_curves"][0]["epoch_count"] = len(plan["epochs"])
        plan["service_model_provenance"]["service_curves"][0]["description_count"] = 6
        plan["summary"]["service_curves"][0]["epoch_count"] = len(plan["epochs"])
        plan["summary"]["service_curves"][0]["description_count"] = 6

        metadata = oracle.identify_service_framework(raw)
        self.assertEqual(metadata["status"], "identified")
        self.assertEqual(metadata["actual_plan_epoch_count"], len(plan["epochs"]))
        coverage = oracle._validate_curve_serialization(raw)
        self.assertEqual(coverage["reconstructed_description_count"], 6)
        oracle._validate_framework_ranges(raw)

    def test_ten_percent_target_passes_and_policy_is_reported(self):
        raw = make_fixture(0.05)
        report = oracle.build_report(raw, entry_parameters={"test": True})
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["schema_version"], 2)
        policy = report["precision_policy"]
        self.assertEqual(policy["policy_id"],
                         "die_accuracy_target_10pct_fallback_20pct_v1")
        self.assertEqual(policy["target_relative_error_limits"]["completion"], 0.10)
        self.assertEqual(policy["fallback_relative_error_limits"]["completion"], 0.20)
        self.assertTrue(policy["target_met"])
        self.assertFalse(policy["target_unmet"])
        self.assertEqual(report["available_granularity_levels"]["status"],
                         "fixture_comparison_only")
        observed = report["available_granularity_levels"]["observed"]
        self.assertEqual(observed[0]["description_count"], 7)
        self.assertEqual(observed[0]["description_budget"], 8)
        self.assertIsNone(report["available_granularity_levels"][
            "minimum_certified_framework"])
        self.assertEqual(report["entry_parameters"], {"test": True})
        self.assertEqual(report["comparisons"]["standalone"]["status"], "passed")
        self.assertEqual(len(report["comparisons"]["two_comp"]), 1)
        self.assertEqual(report["runtime_cumulative_serialization_reconciliation"]["status"],
                         "within_relative_tolerance")
        self.assertTrue(report["runtime_cumulative_serialization_reconciliation"]
                        ["all_available_scenarios_within_tolerance"])

    def test_ten_to_twenty_percent_requires_documented_fallback_evidence(self):
        raw = make_fixture(0.15)
        missing = oracle.build_report(raw)
        self.assertEqual(missing["status"], "failed")
        self.assertTrue(missing["precision_policy"]["target_unmet"])
        self.assertTrue(missing["precision_policy"]["within_fallback_range"])
        self.assertEqual(missing["precision_policy"]["status"],
                         "fallback_not_accepted")
        self.assertFalse(missing["precision_policy"]["fallback_accepted"])
        self.assertEqual(missing["comparisons"]["standalone"]
                         ["precision_assessment"]["status"],
                         "fallback_evidence_missing")

        evidence = {
            "source": "operator spreadsheet breakdown and recorded model checks",
            "attempts": ["rechecked tile-level work accounting", "tested an alternate documented eta setting"],
            "reason": "remaining discrepancy is within the documented 20 percent allowance",
        }
        accepted = oracle.build_report(raw, tolerance_evidence=evidence)
        self.assertEqual(accepted["status"], "accepted_with_fallback")
        self.assertTrue(accepted["precision_policy"]["target_unmet"])
        self.assertTrue(accepted["precision_policy"]["fallback_accepted"])
        self.assertEqual(accepted["precision_policy"]["fallback_evidence"]["source"],
                         evidence["source"])
        self.assertEqual(accepted["comparisons"]["standalone"]
                         ["precision_assessment"]["status"],
                         "accepted_with_fallback")

    def test_over_twenty_percent_and_failed_work_cannot_use_fallback(self):
        evidence = {
            "source": "test source",
            "attempts": ["attempt one", "attempt two"],
            "reason": "documented tolerance record",
        }
        outside = oracle.build_report(make_fixture(0.25), tolerance_evidence=evidence)
        self.assertEqual(outside["status"], "failed")
        self.assertFalse(outside["precision_policy"]["within_fallback_range"])
        self.assertFalse(outside["precision_policy"]["fallback_accepted"])
        self.assertEqual(outside["precision_policy"]["status"],
                         "outside_fallback_range")

        bad_work = make_fixture(0.15)
        bad_work["compiled_plan"]["summary"]["hbm_read_bytes_total"] += 1
        rejected_work = oracle.build_report(bad_work, tolerance_evidence=evidence)
        self.assertEqual(rejected_work["status"], "failed")
        self.assertEqual(rejected_work["compiled_plan_ledger"]["status"], "failed")
        self.assertFalse(rejected_work["precision_policy"]["fallback_accepted"])
        self.assertEqual(rejected_work["precision_policy"]["status"],
                         "work_or_scenario_checks_failed")

        missing_scenario = make_fixture(0.15)
        missing_scenario["two_comp"] = []
        rejected_coverage = oracle.build_report(
            missing_scenario, tolerance_evidence=evidence)
        self.assertEqual(rejected_coverage["status"], "failed")
        self.assertFalse(rejected_coverage["precision_policy"]["fallback_accepted"])

    def test_runtime_serialized_work_failure_blocks_target_and_fallback(self):
        evidence = {
            "source": "recorded spreadsheet and runtime checks",
            "attempts": ["checked the wave ledger", "repeated the serialized counter comparison"],
            "reason": "the remaining timing error is in the documented fallback range",
        }
        raw = make_fixture(0.05)
        raw["standalone"]["counters"][
            "hbm_served_bytes_by_row_direction_exact"][0][0] = "987654321"
        report = oracle.build_report(raw, tolerance_evidence=evidence)
        reconciliation = report["runtime_cumulative_serialization_reconciliation"]
        self.assertEqual(report["compiled_plan_ledger"]["status"], "passed")
        self.assertEqual(reconciliation["status"], "failed")
        self.assertFalse(reconciliation["all_available_scenarios_within_tolerance"])
        self.assertTrue(reconciliation["blocks_acceptance"])
        self.assertEqual(report["status"], "failed")
        self.assertFalse(report["precision_policy"]["fallback_accepted"])
        self.assertEqual(report["precision_policy"]["status"],
                         "runtime_serialized_work_reconciliation_failed")

        incomplete = make_fixture(0.15)
        incomplete["standalone"]["counters"]["accounting_history_complete"] = False
        incomplete_report = oracle.build_report(incomplete, tolerance_evidence=evidence)
        self.assertEqual(incomplete_report[
            "runtime_cumulative_serialization_reconciliation"]["status"], "failed")
        self.assertFalse(incomplete_report["precision_policy"]["fallback_accepted"])

    def test_not_available_serialized_work_stays_unverified_and_distinct(self):
        evidence = {
            "source": "recorded spreadsheet and runtime checks",
            "attempts": ["checked the wave ledger", "repeated the timing comparison"],
            "reason": "the remaining timing error is in the documented fallback range",
        }
        raw = make_fixture(0.15)
        for observation in [raw["standalone"], *raw["two_comp"]]:
            observation["counters"].pop(
                "hbm_served_bytes_by_row_direction_exact")
            observation["counters"].pop("served_by_category")
        report = oracle.build_report(raw, tolerance_evidence=evidence)
        reconciliation = report["runtime_cumulative_serialization_reconciliation"]
        self.assertEqual(report["status"], "accepted_with_fallback")
        self.assertEqual(reconciliation["status"], "not_available")
        self.assertFalse(reconciliation["all_available_scenarios_within_tolerance"])
        self.assertFalse(reconciliation["blocks_acceptance"])
        self.assertTrue(all(item["outcome"] == "not_available"
                            for item in reconciliation["scenarios"]))
        self.assertIn("do not verify exact served work",
                      reconciliation["interpretation"])
        self.assertIn("exact serialized Runtime work counters are unavailable",
                      report["status_reason"])

    def test_missing_or_unknown_reconciliation_status_is_not_a_pass(self):
        summary = oracle._serialized_runtime_work_reconciliation([
            ("standalone", {}),
            ("two_comp[0]", {
                "runtime_cumulative_serialization_reconciliation": {
                    "status": "future_status", "available": True,
                },
            }),
        ])
        self.assertEqual(summary["status"], "failed")
        self.assertTrue(summary["blocks_acceptance"])
        self.assertFalse(summary["all_available_scenarios_within_tolerance"])
        self.assertEqual([item["outcome"] for item in summary["scenarios"]],
                         ["missing", "unrecognized_status"])

    def test_compare_limits_keep_legacy_defaults_and_reject_invalid_values(self):
        raw = make_fixture(0.15)
        waves, expected = oracle._make_discrete_waves(raw)
        solo = oracle.grouped.discrete_grouped(
            waves, expected["true_output_tiles"], [0.0],
            reference.make_capacities(raw))
        legacy = reference.compare_measurement(raw["standalone"], solo, 0.0, False)
        self.assertEqual(legacy["limits"]["completion"], 0.03)
        self.assertEqual(legacy["limits"]["hbm_busy"], 0.05)
        self.assertIsNone(legacy["limits"]["shared_hbm_busy"])

        explicit = reference.compare_measurement(
            raw["standalone"], solo, 0.0, False,
            completion_limit=0.10, busy_limit=0.10)
        self.assertEqual(explicit["limits"]["completion"], 0.10)
        self.assertEqual(explicit["limits"]["hbm_busy"], 0.10)
        self.assertGreater(explicit["relative_errors"]["hbm_busy"], 0.10)

        for invalid in (-0.01, 1.01, float("nan"), float("inf"), True):
            with self.subTest(limit=invalid):
                with self.assertRaises(ValueError):
                    reference.compare_measurement(
                        raw["standalone"], solo, 0.0, False,
                        completion_limit=invalid)
        with self.assertRaisesRegex(ValueError, "busy_limit"):
            reference.compare_measurement(
                raw["standalone"], solo, 0.0, False, busy_limit=-0.1)

    def test_cli_reads_only_a_valid_explicit_evidence_file(self):
        evidence = {
            "source": "documented spreadsheet and runtime checks",
            "attempts": ["first check", "second check"],
            "reason": "remaining error falls inside the 20 percent allowance",
        }
        with tempfile.TemporaryDirectory() as directory:
            temp = Path(directory)
            raw_path = temp / "raw.json"
            evidence_path = temp / "tolerance-evidence.json"
            report_path = temp / "report.json"
            raw_path.write_text(json.dumps(make_fixture(0.15)))
            evidence_path.write_text(json.dumps(evidence))
            result = oracle.main([
                "--raw", str(raw_path),
                "--tolerance-evidence", str(evidence_path),
                "--output", str(report_path),
            ])
            self.assertEqual(result, 0)
            report = json.loads(report_path.read_text())
            self.assertEqual(report["status"], "accepted_with_fallback")
            self.assertEqual(report["entry_parameters"]["tolerance_evidence"],
                             str(evidence_path))

            evidence_path.write_text(json.dumps({
                "source": "one record only", "attempts": ["single attempt"],
                "reason": "not enough history",
            }))
            with self.assertRaisesRegex(ValueError, "at least two"):
                oracle.load_tolerance_evidence(evidence_path)

    def test_incomplete_multicore_and_nonfinite_capacity_are_unsupported(self):
        multicore = make_fixture()
        multicore["synthetic_hardware"]["core_count"] = 2
        with self.assertRaisesRegex(oracle.UnsupportedScenario, "single-Core"):
            oracle.build_report(multicore)

        nonfinite = make_fixture()
        nonfinite["synthetic_hardware"]["resource_capacities"][0]["capacity"] = float("inf")
        with self.assertRaisesRegex(oracle.UnsupportedScenario, "non-finite"):
            oracle.build_report(nonfinite)


if __name__ == "__main__":
    unittest.main()
