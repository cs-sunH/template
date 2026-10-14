#!/usr/bin/env python3
"""Focused JOINT coarse v2 builder/client tests (no native projection claim)."""

from __future__ import annotations

import ctypes
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

_ONLINE_DIR = Path(__file__).resolve().parent
_WORKLOAD_DIR = _ONLINE_DIR.parent
for _path in (_ONLINE_DIR, _WORKLOAD_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import die_model_oracle as oracle_module  # noqa: E402
from die_model_adapter import (  # noqa: E402
    COARSE_DIE_TASK_SCHEMA,
    DieModelContext,
    build_coarse_die_task,
    require_legacy_projection_context,
)
from generate_trace import TraceBuilder  # noqa: E402
from graph_batch_builder import OnlineTraceBuilder  # noqa: E402


def _context(*, coarse: bool = True) -> DieModelContext:
    return DieModelContext(
        profile="coarse_die_v2" if coarse else "mesh_die_v1",
        die_config_path=Path("/not-read-by-builder/resolved_die_model.json"),
        config_hash="ab" * 32,
        dtype="bf16",
        model_dims={
            "hidden": 4096,
            "ffn": 11008,
            "vocab": 32000,
            "mlp_variant": "swiglu",
            "norm_type": "rmsnorm",
            "position_encoding": "rope",
            "num_q_heads": 32,
            "num_kv_heads": 32,
            "head_dim": 128,
        },
        num_q_heads=32,
        num_kv_heads=32,
        ffn_size=11008,
        vocab_size=32000,
        task_schema=(COARSE_DIE_TASK_SCHEMA if coarse
                     else "die_op_spec_v1"),
        execution_profile=("int4_matrix_vec_fp32_assumed_v1"
                           if coarse else None),
    )


class CoarseTaskBuilderTests(unittest.TestCase):
    def test_builds_native_v2_task_from_typed_joint_fields(self) -> None:
        spans = [{
            "request_id_prefix": "decode#span",
            "first_index": 7,
            "run": {
                "q_tokens": 1,
                "kv_first": 100,
                "kv_stride": 1,
                "count": 4,
            },
        }]
        task = build_coarse_die_task(
            "attention_region", _context(), spans,
            layer_count=32,
            weight_passes=1,
            fusion_policy="auto",
            graph_identity={"request_id": "decode-7", "node_id": 4},
            residency={"kv": "local"},
            readiness_scope={"kv": "ready"},
            causal=True,
        )

        self.assertEqual(task["schema"], "coarse_die_task_v2")
        self.assertEqual(task["op_kind"], "attention_region")
        self.assertEqual(task["tensor_shapes"]["m"], 4)
        self.assertEqual(task["tensor_shapes"]["q_heads"], 32)
        self.assertEqual(task["tensor_shapes"]["hidden"], 4096)
        self.assertEqual(task["spans"], spans)
        self.assertEqual(task["storage_dtype"], "bf16")
        self.assertEqual(
            task["execution_profile"], "int4_matrix_vec_fp32_assumed_v1")
        self.assertEqual(task["layer_instances"], 32)
        self.assertEqual(task["weight_passes"], 1)
        self.assertEqual(task["graph_identity"]["node_id"], 4)
        self.assertTrue(task["causal"])
        self.assertNotIn("schema_version", task)
        self.assertNotIn("model_dims", task)
        self.assertNotIn("kv_access", task)

    def test_dtype_roles_and_explicit_task_schema(self) -> None:
        task = build_coarse_die_task(
            "mlp_up", _context(), [{
                "request_id": "prefill#0", "q_tokens": 8,
                "kv_tokens": 8, "q_start_pos": 0, "kv_start_pos": 0,
            }],
            layer_count=2,
            weight_passes=3,
            storage_dtype={"weights": "int4", "activations": "bf16"},
        )
        self.assertEqual(task["storage_dtype"], {
            "weights": "int4", "activations": "bf16"})
        self.assertEqual(task["tensor_shapes"]["m"], 8)

    def test_context_dispatch_and_both_node_wires_preserve_schema(self) -> None:
        context = _context()
        context.begin_emission(
            phase="decode", tensor_parallel=1, tensor_parallel_rank=0,
            span_namespace="wire", pass_spans=[(1, 32)])
        task = context.build_task(
            "norm", layer_count=1, weight_passes=1)
        online = OnlineTraceBuilder(0, remote_operand_loads=False)
        online.comp("typed", 8, 16, die_task=task)
        node = online.nodes[0]
        self.assertEqual(node["compute"]["die_task"], task)
        self.assertNotIn("die_op_spec", node["compute"])

        static = TraceBuilder(remote_operand_loads=False)
        static.comp("typed", 8, 16, die_task=task)
        attrs = {attr.name: attr for attr in static.nodes[0].attr}
        self.assertEqual(json.loads(attrs["die_task"].string_val), task)
        self.assertNotIn("die_op_spec", attrs)

    def test_online_builder_rejects_mixed_or_untyped_payload(self) -> None:
        builder = OnlineTraceBuilder(0, remote_operand_loads=False)
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            builder.comp("mixed", 1, 1, die_spec={}, die_task={
                "schema": "coarse_die_task_v2"})
        with self.assertRaisesRegex(ValueError, "coarse_die_task_v2"):
            builder.comp("bad", 1, 1, die_task={"schema": "unknown"})

    def test_legacy_context_and_untyped_local_region_fail_closed(self) -> None:
        span = [{"request_id": "decode#0", "q_tokens": 1,
                 "kv_tokens": 2, "q_start_pos": 1, "kv_start_pos": 0}]
        with self.assertRaisesRegex(ValueError, "requires a coarse_die_v2"):
            build_coarse_die_task(
                "norm", _context(coarse=False), span,
                layer_count=1, weight_passes=1)
        with self.assertRaisesRegex(ValueError, "unsupported coarse_die_task"):
            build_coarse_die_task(
                "local_region", _context(), span,
                layer_count=1, weight_passes=1)

    def test_empty_or_malformed_spans_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one span"):
            build_coarse_die_task(
                "norm", _context(), [], layer_count=1, weight_passes=1)
        with self.assertRaisesRegex(ValueError, "positive integer"):
            build_coarse_die_task(
                "norm", _context(), [{"q_tokens": 0}],
                layer_count=1, weight_passes=1)

    def test_frozen_legacy_projection_refuses_coarse_context(self) -> None:
        with self.assertRaisesRegex(
                RuntimeError, "frozen native projection has not been validated"):
            require_legacy_projection_context(_context())
        require_legacy_projection_context(_context(coarse=False))

    def test_service_context_requires_matching_hardware_and_system_profile(self) -> None:
        class Config:
            bytes_per_elem = 2
            mlp_variant = "swiglu"
            hidden_size = 4096
            num_heads = 32
            ffn_size = 11008
            vocab_size = 32000
            position_encoding = "rope"
            system_config = None

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            hardware = root / "resolved_die_model.json"
            hardware.write_text(json.dumps({
                "schema": "coarse_die_v2",
                "performance_profile": {
                    "execution_profile": "int4_matrix_vec_fp32_assumed_v1",
                },
            }), encoding="utf-8")
            system = root / "system.json"
            system.write_text(json.dumps({
                "die-model-profile": "coarse_die_v2",
            }), encoding="utf-8")
            Config.system_config = str(system)
            context = DieModelContext.from_service({
                "compute-cost-model": "die",
                "die-model-config": str(hardware),
            }, Config())
            self.assertEqual(context.task_schema, COARSE_DIE_TASK_SCHEMA)
            self.assertEqual(context.execution_profile,
                             "int4_matrix_vec_fp32_assumed_v1")

            system.write_text(json.dumps({"die-model-profile": "mesh_die_v1"}),
                              encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "requires explicit"):
                DieModelContext.from_service({
                    "compute-cost-model": "die",
                    "die-model-config": str(hardware),
                }, Config())


class _FakeFunction:
    def __init__(self, callback):
        self.callback = callback
        self.argtypes = None
        self.restype = None

    def __call__(self, *args):
        return self.callback(*args)


class _FakeCoarseLibrary:
    def __init__(self):
        self.buffers = []
        self.freed = []
        self.destroyed = []
        self.created_config = None
        self.estimate_payload = None
        self.summary_json = {
            "schema_version": "coarse_die_summary_v2",
            "derived_hardware": {"core_count": 88},
        }
        self.result_json = {
            "results": [
                {"estimated_latency_ns": 1234, "plan_hash": "abc"},
                None,
            ],
            "errors": [None, {"reason": "unsupported_dtype"}],
        }
        self.die_coarse_create_model_v2 = _FakeFunction(self._create_model)
        self.die_coarse_hardware_summary_v2 = _FakeFunction(
            lambda _model: self._buffer(json.dumps(self.summary_json)))
        self.die_coarse_estimate_batch_v2 = _FakeFunction(self._estimate)
        self.die_coarse_destroy_model_v2 = _FakeFunction(self._destroy_model)
        self.die_coarse_free_result_v2 = _FakeFunction(self._free)
        self.die_coarse_last_error_v2 = _FakeFunction(lambda: b"fake error")

    def _buffer(self, payload: str) -> int:
        buffer = ctypes.create_string_buffer(payload.encode("utf-8"))
        self.buffers.append(buffer)
        return ctypes.cast(buffer, ctypes.c_void_p).value

    def _create_model(self, config_json, model_out) -> int:
        self.created_config = config_json.decode("utf-8")
        ctypes.cast(model_out, ctypes.POINTER(ctypes.c_uint64))[0] = 17
        return 0

    def _estimate(self, _model, payload) -> int:
        self.estimate_payload = json.loads(payload.decode("utf-8"))
        return self._buffer(json.dumps(self.result_json))

    def _destroy_model(self, model) -> int:
        self.destroyed.append(model)
        return 0

    def _free(self, handle) -> None:
        self.freed.append(handle)


class CoarseOracleTests(unittest.TestCase):
    def test_coarse_v2_abi_is_selected_and_legacy_execution_is_blocked(self) -> None:
        fake = _FakeCoarseLibrary()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = root / "hardware.json"
            config.write_text(json.dumps({"schema": "coarse_die_v2"}),
                              encoding="utf-8")
            library_path = root / "libcoarse_die_cabi.so"
            library_path.touch()
            with mock.patch.object(
                    oracle_module, "discover_coarse_cabi_library",
                    return_value=library_path), mock.patch.object(
                    ctypes, "CDLL", return_value=fake):
                oracle = oracle_module.DieModelOracle(
                    config_json_path=str(config))
                try:
                    self.assertTrue(oracle.coarse)
                    self.assertEqual(oracle.hardware_summary()["derived"],
                                     {"core_count": 88})
                    valid = {"schema": "coarse_die_task_v2",
                             "op_kind": "norm"}
                    result = oracle.estimate([valid, {"bad": True}])
                    self.assertEqual(result[0]["estimated_latency_ns"], 1234)
                    self.assertIsNone(result[1])
                    self.assertEqual(oracle.last_errors,
                                     [None, "unsupported_dtype"])
                    self.assertEqual(fake.estimate_payload,
                                     {"specs": [valid, {"bad": True}]})
                    with self.assertRaisesRegex(
                            RuntimeError, "unsupported_projection_state"):
                        oracle.create_execution(valid)
                    self.assertEqual(len(fake.freed), 2)
                finally:
                    oracle.close()
                    oracle.close()
            self.assertEqual(fake.destroyed, [17])

    def test_config_schema_must_select_v2_abi(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / "hardware.json"
            config.write_text('{"schema":"coarse_die_v2"}',
                              encoding="utf-8")
            with mock.patch.object(
                    oracle_module, "discover_coarse_cabi_library",
                    side_effect=RuntimeError("coarse ABI absent")):
                with self.assertRaisesRegex(RuntimeError, "coarse ABI absent"):
                    oracle_module.DieModelOracle(
                        config_json_path=str(config))


if __name__ == "__main__":
    unittest.main()
