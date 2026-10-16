#!/usr/bin/env python3
"""test_die_model_oracle.py -- 阶段 D die 成本 oracle(ctypes 客户端)测试。

License: MIT (see LICENSE in the repository root).

覆盖(任务书 §4):
  - 发现序与缺库 fail-closed(绝不静默回退 roofline);
  - resolved_die_model.json 缺位 fail-closed;
  - estimate 批量 + 坏 spec 错误位(好位不受坏位影响);
  - hardware_summary 派生块(face_case5 die 点:explicit 23x23/529 Cores/
    669.714 INT4 TOPS compatibility scalar/1.64e12 B/s);
  - 同一 spec 两次估值确定(逐字段相等);
  - config 路径解析序(显式 → env → 仓内发现 → manifest);
  - close() 生命周期。

运行前需先构建共享库并导出发现序(或落在仓内 build 路径):
    cmake -S extern/die_model -B /tmp/die_build/stage_D/cabi_build \
        && cmake --build /tmp/die_build/stage_D/cabi_build -j
    export DIE_MODEL_CABI_PATH=/tmp/die_build/stage_D/lib/libdie_model_cabi.so
库缺位时本文件跳过(legacy 回归套件保持全绿);oracle 自身在
backend=die 的生产路径上对缺库 fail-closed,与测试跳过语义不同。
"""

from __future__ import annotations

import ctypes
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


MODULE_DIR = Path(__file__).resolve().parent
ONLINE_DIR = MODULE_DIR / "online"
for _path in (str(MODULE_DIR), str(ONLINE_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from die_model_oracle import (  # noqa: E402
    CABI_LIBRARY_ENV,
    COARSE_CABI_LIBRARY_ENV,
    COST_RESULT_FIELDS,
    DIE_MODEL_CONFIG_ENV,
    MANIFEST_COMPUTE_COST_MODEL_KEY,
    MANIFEST_DIE_MODEL_CONFIG_KEY,
    DieModelOracle,
    discover_cabi_library,
    resolve_die_model_config_path,
)

# face_case5 die 点(契约 §3.5 显式fixture:23×23 / 0.75 MiB SRAM / 800 mm² /
# noc 512 bit / HBM 4x1 单侧)。wrapped resolved_die_model.json 形态
# (契约 §3.3;derived 由库派生,回灌可忽略)。
FACE_CASE5_DIE_CONFIG = {
    "schema_version": 1,
    "profile": "mesh_die_v1",
    "die": {
        "sram-mb-per-core": 0.75,
        "mesh-nx": 23, "mesh-ny": 23,
        "compute-area-mm2": 800,
        "hbm-per-side": 4,
        "hbm-sides": 1,
        "noc-width-bits": 512,
    },
}


def _norm_spec() -> dict:
    """最小合规 norm spec(契约 §2;llama2_7b 维度,tp=1)。"""
    return {
        "schema_version": 1,
        "op_kind": "norm",
        "phase": "prefill",
        "dtype": "bf16",
        "model_dims": {
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
        "tp_slice": {
            "tensor_parallel": 1,
            "tensor_parallel_rank": 0,
            "q_heads_local": 32,
            "kv_heads_local": 32,
            "ffn_local": 11008,
            "vocab_local": 32000,
        },
        "spans": [{
            "request_id": "r0",
            "q_tokens": 128,
            "kv_tokens": 128,
            "q_start_pos": 0,
            "kv_start_pos": 0,
        }],
        "layer_count": 2,
        "weight_passes": 1,
        "kv_access": {"identity": "local"},
        "fusion_policy": "auto",
        "provenance": "test_die_model_oracle",
        "config_hash": "test",
    }


def _library_available() -> bool:
    # 与生产发现序完全一致(env DIE_MODEL_CABI_PATH -> 仓内 build 路径,
    # 契约 §7 裁定 1):不得 mock 掉仓内候选——否则 .so 已落在仓内
    # build/ 发现路径上时本守卫仍会误判缺库而 skip(阶段 E 修复,与
    # test_face_scheduler._die_library_available 同款)。
    try:
        discover_cabi_library(None)
    except RuntimeError:
        return False
    return True


_LIBRARY_READY = None


def library_ready() -> bool:
    global _LIBRARY_READY
    if _LIBRARY_READY is None:
        _LIBRARY_READY = _library_available()
    return _LIBRARY_READY


class _TempConfigMixin:
    """每类一个临时 resolved_die_model.json(避免测试间 env 串扰)。"""

    def setUp(self) -> None:
        self._temp = tempfile.TemporaryDirectory(prefix="die_oracle_test_")
        self.addCleanup(self._temp.cleanup)
        self.config_path = Path(self._temp.name) / "resolved_die_model.json"
        self.config_path.write_text(
            json.dumps(FACE_CASE5_DIE_CONFIG), encoding="utf-8")

    def _make_oracle(self) -> DieModelOracle:
        return DieModelOracle(config_json_path=str(self.config_path))


@unittest.skipUnless(
    library_ready(), "libdie_model_cabi.so not found; build extern/die_model "
    "and export DIE_MODEL_CABI_PATH (see module docstring)")
class DieModelOracleTests(_TempConfigMixin, unittest.TestCase):
    def test_hardware_summary_face_case5_die_point(self) -> None:
        with self._make_oracle() as oracle:
            summary = oracle.hardware_summary()
            derived = summary["derived"]
            self.assertEqual(summary["schema_version"], 1)
            self.assertEqual(summary["profile"], "mesh_die_v1")
            self.assertEqual(derived["mesh_n"], 23)
            self.assertEqual(derived["cores"], 529)
            self.assertAlmostEqual(derived["pe_peak_tflops"], 669.714)
            self.assertAlmostEqual(derived["hbm_bandwidth_bytes"], 1.64e12)
            # 摘要缓存:两次调用同一对象且非空(构造期自检的读取路径)。
            self.assertIs(oracle.hardware_summary(), summary)

    def test_config_hash_is_stable_and_short(self) -> None:
        with self._make_oracle() as first:
            with self._make_oracle() as second:
                self.assertEqual(first.config_hash, second.config_hash)
                self.assertEqual(len(first.config_hash), 16)
                int(first.config_hash, 16)  # 十六进制串

    def test_estimate_batch_and_error_positions(self) -> None:
        good = _norm_spec()
        bad = dict(good)
        bad["op_kind"] = "attention_region"
        bad["spans"] = []  # spans 至少一条 -> 库侧逐条报错
        with self._make_oracle() as oracle:
            results = oracle.estimate([good, bad, good])
            self.assertEqual(len(results), 3)
            self.assertIsNotNone(results[0])
            self.assertIsNone(results[1])  # 坏 spec 不炸批
            self.assertIsNotNone(results[2])
            self.assertIsNone(oracle.last_errors[0])
            self.assertIn("spans", oracle.last_errors[1])
            for result in (results[0], results[2]):
                self.assertEqual(set(result), COST_RESULT_FIELDS)
                self.assertGreater(result["total_seconds"], 0.0)
                self.assertEqual(result["profile"], "mesh_die_v1")

    def test_estimate_is_deterministic_for_same_spec(self) -> None:
        spec = _norm_spec()
        with self._make_oracle() as oracle:
            first = oracle.estimate([spec])[0]
            second = oracle.estimate([spec])[0]
            third = oracle.estimate([dict(spec)])[0]
            self.assertEqual(first, second)
            self.assertEqual(first, third)

    def test_estimate_bad_envelope_fails_closed(self) -> None:
        # 坏 spec 落在批内是逐条错误位路径(上方用例);批量级失败 =
        # 入参信封畸形。estimate() 自身恒构造合规信封,故此处经 oracle
        # 持有的原生句柄直接打畸形信封(白盒),钉住 ABI 错误取回链路。
        with self._make_oracle() as oracle:
            handle = oracle._lib.die_estimate_batch(
                oracle._model, b"{not json")
            self.assertFalse(handle)
            self.assertIn("die_estimate_batch", oracle._last_error())
            # 合规调用仍可用(错误槽不清模型)。
            self.assertIsNotNone(oracle.estimate([_norm_spec()])[0])

    def test_estimate_one_raises_on_rejected_spec(self) -> None:
        bad = _norm_spec()
        bad["layer_count"] = 0
        with self._make_oracle() as oracle:
            with self.assertRaisesRegex(RuntimeError, "rejected"):
                oracle.estimate_one(bad)

    def test_close_is_idempotent_and_guards_queries(self) -> None:
        oracle = self._make_oracle()
        oracle.close()
        oracle.close()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            oracle.hardware_summary()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            oracle.estimate([_norm_spec()])

    def test_bad_config_fails_closed_with_library_error(self) -> None:
        broken = Path(self._temp.name) / "broken_die_model.json"
        broken.write_text(json.dumps({
            "schema_version": 1, "profile": "mesh_die_v1",
            "die": {"sram-mb-per-core": 0.75, "mesh-nx": 23, "mesh-ny": 23,
                    "compute-area-mm2": 7000, "hbm-per-side": 4,
                    "hbm-sides": 1, "noc-width-bits": 512},
        }), encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "die_create_model"):
            DieModelOracle(config_json_path=str(broken))


class CoarseAvailabilityTests(unittest.TestCase):
    """Exercise new statuses without requiring a newer native SDK."""

    def _oracle(self, rows, errors=None, *, coarse=True, supports_cost=True):
        payload = {"results": rows, "errors": errors or [None] * len(rows)}
        buffer = ctypes.create_string_buffer(json.dumps(payload).encode("utf-8"))
        library = mock.Mock()
        library._result_buffer = buffer  # Keep the simulated C allocation alive.
        pointer = ctypes.addressof(buffer)
        library.die_coarse_estimate_batch_v2.return_value = pointer
        library.die_coarse_estimate_cost_batch_v2.return_value = pointer
        library.die_estimate_batch.return_value = pointer
        oracle = DieModelOracle.__new__(DieModelOracle)
        oracle._lib = library
        oracle._model = 17
        oracle._closed = False
        oracle._hardware_summary = {"derived": {"test": 1}}
        oracle.coarse = coarse
        oracle.supports_estimate_cost = supports_cost
        oracle.last_errors = []
        self.addCleanup(oracle.close)
        return oracle, library

    @staticmethod
    def _row(status="ok_assumed"):
        return {
            "estimated_latency_ns": 1234,
            "plan_hash": "selected-plan",
            "status": status,
            "calibration_status": "assumed",
            "heldout_status": "not_available",
        }

    def test_cost_full_and_optional_fallback_keep_available_siblings(self):
        for mode in ("full", "cost", "full_fallback"):
            with self.subTest(mode=mode):
                good = self._row()
                unavailable = self._row("not_available")
                unavailable["prediction_scope"] = "body_timing_only"
                oracle, library = self._oracle(
                    [good, unavailable, None, good],
                    [None, None, {"reason": "unsupported_dtype"}, None],
                    supports_cost=mode != "full_fallback")
                query = oracle.estimate if mode == "full" else oracle.estimate_cost
                result = query([{"op_kind": "norm"}] * 4)
                self.assertEqual(len(result), 4)
                self.assertEqual(result[0]["estimated_latency_ns"], 1234)
                self.assertEqual(result[0], result[3])
                self.assertIsNone(result[1])
                self.assertIsNone(result[2])
                self.assertIn("not_available", oracle.last_errors[1])
                self.assertEqual(oracle.last_errors[2], "unsupported_dtype")
                self.assertIsNone(oracle.last_errors[0])
                self.assertIsNone(oracle.last_errors[3])
                library.die_coarse_free_result_v2.assert_called_once()
                self.assertEqual(library.die_coarse_estimate_cost_batch_v2.call_count,
                                 int(mode == "cost"))
                self.assertEqual(library.die_coarse_estimate_batch_v2.call_count,
                                 int(mode != "cost"))

    def test_unavailable_body_is_rejected_before_latency_conversion(self):
        for cost_only in (False, True):
            with self.subTest(cost_only=cost_only):
                row = self._row("not_available")
                row["estimated_latency_ns"] = "diagnostic body only"
                oracle, library = self._oracle([row])
                query = oracle.estimate_cost if cost_only else oracle.estimate
                self.assertEqual(query([{}]), [None])
                self.assertIn("not_available", oracle.last_errors[0])
                library.die_coarse_free_result_v2.assert_called_once()

    def test_coarse_top_level_status_must_be_a_string(self):
        for mode in ("full", "cost", "full_fallback"):
            for status in (None, True, 7, [], {}):
                with self.subTest(mode=mode, status=status):
                    row = self._row(status)
                    oracle, library = self._oracle(
                        [row], supports_cost=mode != "full_fallback")
                    query = oracle.estimate if mode == "full" else oracle.estimate_cost
                    with self.assertRaisesRegex(RuntimeError, "status"):
                        query([{}])
                    library.die_coarse_free_result_v2.assert_called_once()
            with self.subTest(mode=mode, status="missing"):
                row = self._row()
                del row["status"]
                oracle, library = self._oracle(
                    [row], supports_cost=mode != "full_fallback")
                query = oracle.estimate if mode == "full" else oracle.estimate_cost
                with self.assertRaisesRegex(RuntimeError, "status"):
                    query([{}])
                library.die_coarse_free_result_v2.assert_called_once()

    def test_estimate_one_rejects_unavailable_body(self):
        oracle, _ = self._oracle([self._row("not_available")])
        with self.assertRaisesRegex(RuntimeError, "rejected.*not_available"):
            oracle.estimate_one({"op_kind": "norm"})

    def test_legacy_cost_result_keeps_existing_status_semantics(self):
        row = {key: 0 for key in COST_RESULT_FIELDS}
        row["status"] = "not_available"
        oracle, library = self._oracle([row], coarse=False)
        self.assertEqual(oracle.estimate_cost([{}]), [row])
        self.assertEqual(oracle.last_errors, [None])
        library.die_free_result.assert_called_once()
        library.die_coarse_free_result_v2.assert_not_called()

    def test_face_scorer_rejects_none_without_cost_fallback(self):
        import face_scheduler as scheduler
        model = scheduler.FaceModel(1, 8, 16, 4, 32, 2, "gelu")
        hardware = scheduler.FaceHardware(3, 2, 100, 1.0, 2.0, 1.0, 0, 0)
        arguments = dict(instance_size=2, p_chunk=8, d_batch=1, d_token=257)
        specs, _ = scheduler.build_die_estimate_specs(
            model, config_hash="test", **arguments)
        for mode in ("full", "cost"):
            with self.subTest(mode=mode):
                oracle, library = self._oracle(
                    [self._row("not_available")] * len(specs),
                    supports_cost=mode == "cost")
                with self.assertRaisesRegex(RuntimeError, "rejected.*not_available"):
                    scheduler.estimate_iteration_time_ns(
                        hardware, model, cost_oracle=oracle, **arguments)
                self.assertEqual(library.die_coarse_estimate_batch_v2.call_count,
                                 int(mode == "full"))
                self.assertEqual(library.die_coarse_estimate_cost_batch_v2.call_count,
                                 int(mode == "cost"))

    def test_face_scorer_propagates_oracle_exception(self):
        import face_scheduler as scheduler
        oracle, library = self._oracle([])
        with mock.patch.object(oracle, "estimate_cost",
                               side_effect=RuntimeError("batch transport failed")):
            with self.assertRaisesRegex(RuntimeError, "batch transport failed"):
                scheduler.estimate_iteration_time_ns(
                    scheduler.FaceHardware(3, 2, 100, 1.0, 2.0, 1.0, 0, 0),
                    scheduler.FaceModel(1, 8, 16, 4, 32, 2, "gelu"),
                    cost_oracle=oracle, instance_size=2, p_chunk=8,
                    d_batch=1, d_token=257)
        library.die_coarse_estimate_batch_v2.assert_not_called()
        library.die_estimate_batch.assert_not_called()
@unittest.skipUnless(
    os.environ.get(COARSE_CABI_LIBRARY_ENV),
    "coarse DIE ABI not supplied; set DIE_COARSE_CABI_PATH")
class CoarseCostQueryTests(unittest.TestCase):
    def test_cost_only_batch_keeps_one_ordered_result_per_input(self) -> None:
        repo_root = MODULE_DIR.parents[2]
        config_path = (repo_root / "extern/die_model/coarse/configs"
                       / "example_hardware.json")
        tasks_path = (repo_root / "extern/die_model/coarse/configs"
                      / "example_tasks.json")
        task_document = json.loads(tasks_path.read_text(encoding="utf-8"))
        template = next(
            row for row in task_document["specs"]
            if isinstance(row, dict)
            and row.get("schema") == "coarse_die_task_v2")
        specs = []
        for m in range(1, 8):
            spec = json.loads(json.dumps(template))
            spec["tensor_shapes"]["m"] = m
            specs.append(spec)

        with DieModelOracle(
                lib_path=os.environ[COARSE_CABI_LIBRARY_ENV],
                config_json_path=str(config_path)) as oracle:
            self.assertTrue(oracle.coarse)
            self.assertTrue(oracle.supports_estimate_cost)

            full = oracle.estimate(specs)
            self.assertEqual(len(full), len(specs))
            self.assertEqual(oracle.last_errors, [None] * len(specs))

            selected = oracle.estimate_cost(specs)
            self.assertEqual(len(selected), len(specs))
            self.assertEqual(oracle.last_errors, [None] * len(specs))
            individual = []
            for spec in specs:
                one = oracle.estimate_cost([spec])
                self.assertEqual(len(one), 1)
                self.assertEqual(oracle.last_errors, [None])
                individual.append(one[0])

            for actual, expected, single in zip(selected, full, individual):
                self.assertIsNotNone(actual)
                self.assertIsNotNone(expected)
                self.assertIsNotNone(single)
                self.assertEqual(set(actual), {
                    "estimated_latency_ns", "plan_hash", "status",
                    "total_seconds",
                })
                self.assertEqual(
                    actual["estimated_latency_ns"],
                    expected["estimated_latency_ns"])
                self.assertEqual(actual["plan_hash"],
                                 expected["plan_fingerprint"])
                self.assertEqual(actual["status"], expected["status"])
                self.assertEqual(
                    actual["total_seconds"],
                    float(actual["estimated_latency_ns"]) / 1e9)
                self.assertEqual(actual, single)

            # Simulate an older coarse library that has no additive cost
            # endpoint: estimate_cost must preserve the established full-row
            # fallback, including one result per input and input ordering.
            with mock.patch.object(oracle, "supports_estimate_cost", False):
                fallback = oracle.estimate_cost(specs)
            self.assertEqual(len(fallback), len(specs))
            self.assertEqual(fallback, full)


class DieModelOracleDiscoveryTests(unittest.TestCase):
    """发现序/缺库 fail-closed(不需要库存在;候选表打桩为空)。"""

    def test_missing_library_fails_closed_without_roofline_fallback(self):
        saved = os.environ.pop(CABI_LIBRARY_ENV, None)
        try:
            with mock.patch("die_model_oracle.cabi_library_candidates",
                            return_value=()):
                with self.assertRaisesRegex(
                        RuntimeError, "die_model_cabi.*build"):
                    discover_cabi_library(None)
                # 显式路径缺位同样 fail-closed,报错列出尝试过的候选。
                with self.assertRaisesRegex(RuntimeError, "bogus"):
                    discover_cabi_library("/nonexistent/bogus.so")
        finally:
            if saved is not None:
                os.environ[CABI_LIBRARY_ENV] = saved

    def test_explicit_lib_path_wins_over_env(self):
        env_target = Path(self._temp_copy("env_target.so"))
        explicit_target = Path(self._temp_copy("explicit_target.so"))
        with mock.patch("die_model_oracle.cabi_library_candidates",
                        return_value=()):
            saved = os.environ.get(CABI_LIBRARY_ENV)
            os.environ[CABI_LIBRARY_ENV] = str(env_target)
            try:
                self.assertEqual(
                    discover_cabi_library(str(explicit_target)),
                    explicit_target)
                self.assertEqual(discover_cabi_library(None), env_target)
            finally:
                if saved is None:
                    os.environ.pop(CABI_LIBRARY_ENV, None)
                else:
                    os.environ[CABI_LIBRARY_ENV] = saved

    def test_missing_config_path_fails_closed(self):
        if not library_ready():
            self.skipTest("library not available for the config guard test")
        saved = os.environ.pop(DIE_MODEL_CONFIG_ENV, None)
        try:
            with self.assertRaisesRegex(
                    RuntimeError, "resolved_die_model.json"):
                DieModelOracle()
        finally:
            if saved is not None:
                os.environ[DIE_MODEL_CONFIG_ENV] = saved

    @staticmethod
    def _temp_copy(name: str) -> str:
        directory = tempfile.mkdtemp(prefix="die_oracle_discovery_")
        target = Path(directory) / name
        target.write_bytes(b"placeholder")
        return str(target)


class ResolveConfigPathOrderTests(unittest.TestCase):
    def test_order_explicit_env_repo_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repo_copy = root / "sh_test_mesh" / "generated"
            repo_copy.mkdir(parents=True)
            (repo_copy / "resolved_die_model.json").write_text(
                json.dumps(FACE_CASE5_DIE_CONFIG), encoding="utf-8")
            explicit = root / "explicit.json"
            env_value = root / "env.json"
            manifest = {
                MANIFEST_COMPUTE_COST_MODEL_KEY: "die",
                MANIFEST_DIE_MODEL_CONFIG_KEY: str(root / "manifest.json"),
            }
            for candidate in (explicit, env_value,
                              Path(manifest[MANIFEST_DIE_MODEL_CONFIG_KEY])):
                candidate.write_text(
                    json.dumps(FACE_CASE5_DIE_CONFIG), encoding="utf-8")
            saved = os.environ.pop(DIE_MODEL_CONFIG_ENV, None)
            try:
                # 显式 > env。
                self.assertEqual(
                    resolve_die_model_config_path(
                        explicit=str(explicit), manifest=None, repo_root=root),
                    explicit)
                os.environ[DIE_MODEL_CONFIG_ENV] = str(env_value)
                # env > 仓内发现。
                self.assertEqual(
                    resolve_die_model_config_path(
                        explicit=None, manifest=manifest, repo_root=root),
                    env_value)
                os.environ.pop(DIE_MODEL_CONFIG_ENV, None)
                # 仓内发现 > manifest。
                self.assertEqual(
                    resolve_die_model_config_path(
                        explicit=None, manifest=manifest, repo_root=root),
                    repo_copy / "resolved_die_model.json")
                # 无仓内候选 -> manifest 绝对路径兜底。
                (repo_copy / "resolved_die_model.json").unlink()
                self.assertEqual(
                    resolve_die_model_config_path(
                        explicit=None, manifest=manifest, repo_root=root),
                    Path(manifest[MANIFEST_DIE_MODEL_CONFIG_KEY]))
                # 全缺 -> None(backend=die 时由调用方 fail-closed)。
                self.assertIsNone(
                    resolve_die_model_config_path(
                        explicit=None, manifest={}, repo_root=root))
            finally:
                if saved is not None:
                    os.environ[DIE_MODEL_CONFIG_ENV] = saved


if __name__ == "__main__":
    unittest.main()
