#!/usr/bin/env python3
"""Typed Die hardware adapter contract for the LRU WSC-LLM loader."""
import os
import sys
import unittest
from types import SimpleNamespace

_WORKLOAD_DIR = os.path.dirname(os.path.abspath(__file__))
_SH_TEST_DIR = os.path.dirname(os.path.dirname(_WORKLOAD_DIR))
for _path in (_WORKLOAD_DIR, _SH_TEST_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from generate_wsc_llm_trace import _to_wsc_llm_hardware  # noqa: E402
from wsc_llm_scheduler import WscLlmHardware  # noqa: E402


_CAPACITY_BYTES = 144_000_000_000
_UINT64_MAX = (1 << 64) - 1
_INT4_OPS_PER_S = 769_296_000_000_000.0
_VEC_EQOPS_PER_S = 76_924_619_200_000.0


def _resolved_coarse_hardware(**overrides):
    values = {
        "mesh_rows": 1,
        "mesh_cols": 2,
        "local_hbm_capacity_bytes": _CAPACITY_BYTES,
        "local_hbm_bandwidth_gbps": 4_710.4,
        "local_hbm_latency_ns": 100,
        "d2d_bandwidth_gbps": 4_050.0,
        "d2d_latency_ns": 5,
        "peak_perf_tflops": None,
        "label": "tiny coarse hardware",
        "die_profile": "coarse_die_v2",
        "die_derived": {
            "hardware_summary": {
                "peak_int4_ops_per_s": _INT4_OPS_PER_S,
                "peak_vec_fp32_eq_ops_per_s": _VEC_EQOPS_PER_S,
                "hbm_row_peak_Bps": [2_355_200_000_000.0] * 2,
                "hbm_capacity_bytes": _CAPACITY_BYTES,
            },
        },
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class WscLlmTypedHardwareTests(unittest.TestCase):
    def test_coarse_adapter_preserves_typed_rates_and_exact_capacity(self):
        hardware = _to_wsc_llm_hardware(_resolved_coarse_hardware())

        self.assertEqual(hardware.die_profile, "coarse_die_v2")
        self.assertEqual(hardware.die_peak_int4_ops_s, _INT4_OPS_PER_S)
        self.assertEqual(hardware.die_peak_vec_eqops_s, _VEC_EQOPS_PER_S)
        self.assertIsNone(hardware.peak_perf_tflops)
        self.assertEqual(hardware.local_hbm_capacity_bytes, _CAPACITY_BYTES)
        self.assertIs(type(hardware.local_hbm_capacity_bytes), int)
        self.assertEqual(hardware.local_hbm_bandwidth_gbps, 4_710.4)
        self.assertEqual(hardware.d2d_bandwidth_gbps, 4_050.0)
        self.assertEqual(hardware.d2d_latency_ns, 5)
        self.assertEqual(hardware.local_hbm_latency_ns, 100)
        with self.assertRaisesRegex(ValueError, "no generic peak_perf_tflops"):
            hardware.require_legacy_peak_perf_tflops()

    def test_adapter_preserves_uint64_capacity_without_float_rounding(self):
        hardware = _to_wsc_llm_hardware(
            _resolved_coarse_hardware(local_hbm_capacity_bytes=_UINT64_MAX))

        self.assertEqual(hardware.local_hbm_capacity_bytes, _UINT64_MAX)
        self.assertIs(type(hardware.local_hbm_capacity_bytes), int)

    def test_legacy_scalar_remains_valid_and_guarded_getter_returns_it(self):
        resolved = SimpleNamespace(
            mesh_rows=1,
            mesh_cols=2,
            local_hbm_capacity_bytes=64_000_000_000,
            local_hbm_bandwidth_gbps=1_640.0,
            local_hbm_latency_ns=100,
            d2d_bandwidth_gbps=4_050.0,
            d2d_latency_ns=5,
            peak_perf_tflops=1_492.05,
            label="legacy mesh hardware",
            die_profile="mesh_die_v1",
            die_derived=None,
        )
        hardware = _to_wsc_llm_hardware(resolved)
        self.assertEqual(hardware.peak_perf_tflops, 1_492.05)
        self.assertEqual(hardware.require_legacy_peak_perf_tflops(), 1_492.05)

    def test_coarse_compatibility_scalar_never_becomes_generic_getter(self):
        hardware = _to_wsc_llm_hardware(
            _resolved_coarse_hardware(peak_perf_tflops=2.5))
        self.assertEqual(hardware.peak_perf_tflops, 2.5)
        with self.assertRaisesRegex(ValueError, "typed INT4/FP32-equivalent"):
            hardware.require_legacy_peak_perf_tflops()

    def test_missing_or_invalid_coarse_typed_rates_fail_closed(self):
        for field, value in (
                ("peak_int4_ops_per_s", None),
                ("peak_int4_ops_per_s", True),
                ("peak_int4_ops_per_s", 0.0),
                ("peak_vec_fp32_eq_ops_per_s", float("inf")),
                ("peak_vec_fp32_eq_ops_per_s", float("nan"))):
            with self.subTest(field=field, value=value):
                resolved = _resolved_coarse_hardware()
                summary = dict(resolved.die_derived["hardware_summary"])
                summary[field] = value
                resolved.die_derived = {"hardware_summary": summary}
                with self.assertRaisesRegex(ValueError, "finite positive rate"):
                    _to_wsc_llm_hardware(resolved)

    def test_coarse_profile_needs_both_finite_positive_typed_rates(self):
        for field, value in (
                ("die_peak_int4_ops_s", None),
                ("die_peak_int4_ops_s", True),
                ("die_peak_int4_ops_s", float("inf")),
                ("die_peak_vec_eqops_s", 0.0),
                ("die_peak_vec_eqops_s", float("nan"))):
            with self.subTest(field=field, value=value):
                values = {
                    "mesh_rows": 1,
                    "mesh_cols": 2,
                    "local_hbm_capacity_bytes": _CAPACITY_BYTES,
                    "local_hbm_bandwidth_gbps": 4_710.4,
                    "d2d_bandwidth_gbps": 4_050.0,
                    "peak_perf_tflops": None,
                    "d2d_latency_ns": 5,
                    "local_hbm_latency_ns": 100,
                    "die_profile": "coarse_die_v2",
                    "die_peak_int4_ops_s": _INT4_OPS_PER_S,
                    "die_peak_vec_eqops_s": _VEC_EQOPS_PER_S,
                }
                values[field] = value
                with self.assertRaisesRegex(ValueError, "finite positive rate"):
                    WscLlmHardware(**values)


if __name__ == "__main__":
    unittest.main()
