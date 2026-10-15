#!/usr/bin/env python3
"""Focused state/provenance tests for the offline analytical service oracle."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "validation"))

import analytical_service_oracle as oracle
import compiled_gemm_oracle as reference


def make_fixture() -> dict:
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
    return raw


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

    def test_failed_scenario_is_reported_as_precision_failure(self):
        raw = make_fixture()
        report = oracle.build_report(raw, entry_parameters={"test": True})
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["available_granularity_levels"]["status"],
                         "fixture_comparison_only")
        observed = report["available_granularity_levels"]["observed"]
        self.assertEqual(observed[0]["description_count"], 7)
        self.assertEqual(observed[0]["description_budget"], 8)
        self.assertIsNone(report["available_granularity_levels"][
            "minimum_certified_framework"])
        self.assertEqual(report["entry_parameters"], {"test": True})
        self.assertEqual(report["comparisons"]["standalone"]["status"], "failed")

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
