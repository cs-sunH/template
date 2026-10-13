from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest


_SH_TEST_DIR = Path(__file__).resolve().parents[1]
if str(_SH_TEST_DIR) not in sys.path:
    sys.path.insert(0, str(_SH_TEST_DIR))

from config_resolver import (
    RuntimeConfigPaths,
    _MANAGED_SYSTEM_FIELDS,
    cabi_library_candidates,
    discover_cabi_library,
    load_hardware_config,
    materialize_runtime_configs,
)


_HARDWARE_SOURCE = _SH_TEST_DIR / "hardware" / "face_case5_config_c.json"
_SYSTEM_TEMPLATE = _SH_TEST_DIR / "system" / "llama2_7b_roofline_template.json"
_DIE_SYSTEM_TEMPLATE = _SH_TEST_DIR / "system" / "llama2_7b_die_template.json"

# die 推广第一期（2026-10-03）：checked-in 硬件默认切换为 die 节。legacy
# 路径的既有断言经"换回 compute 节"的临时硬件拷贝全部保留（断言原文不
# 动，只换装载源）；die 节路径由 DieHardwareTests 覆盖（checked-in 源）。
_LEGACY_COMPUTE_NODE = {"peak-perf-tflops": 261.12}


def _legacy_hardware_copy(directory: Path) -> Path:
    source = json.loads(_HARDWARE_SOURCE.read_text(encoding="utf-8"))
    del source["die"]
    source["compute"] = dict(_LEGACY_COMPUTE_NODE)
    path = Path(directory) / "legacy_hardware.json"
    path.write_text(json.dumps(source, indent=2) + "\n", encoding="utf-8")
    return path


class ConfigResolverTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self._temporary.cleanup)
        self.hardware_path = _legacy_hardware_copy(Path(self._temporary.name))
        self.source = json.loads(self.hardware_path.read_text(encoding="utf-8"))
        self.profile_name = next(iter(self.source["local-hbm"]["capacity-profiles"]))
        self.profile = self.source["local-hbm"]["capacity-profiles"][self.profile_name]
        self.hardware = load_hardware_config(self.hardware_path, self.profile_name)

    def test_resolves_values_and_profile_from_the_single_hardware_source(self) -> None:
        mesh = self.source["mesh"]
        local_hbm = self.source["local-hbm"]
        self.assertEqual(self.hardware.slug, self.source["slug"])
        self.assertEqual(self.hardware.mesh_rows, mesh["rows"])
        self.assertEqual(self.hardware.mesh_cols, mesh["columns"])
        self.assertEqual(self.hardware.npus_count, mesh["rows"] * mesh["columns"])
        self.assertEqual(
            self.hardware.topology_by_network_dimension,
            tuple(mesh["topology-by-network-dimension"]),
        )
        self.assertEqual(self.hardware.local_hbm_capacity_bytes, self.profile["bytes"])
        self.assertEqual(self.hardware.local_hbm_bandwidth_gbps, local_hbm["bandwidth-gbps"])
        self.assertEqual(self.hardware.local_hbm_latency_ns, local_hbm["latency-ns"])
        self.assertEqual(self.hardware.d2d_bandwidth_gbps, self.source["d2d"]["bandwidth-gbps"])
        self.assertEqual(self.hardware.d2d_latency_ns, self.source["d2d"]["latency-ns"])
        self.assertEqual(
            self.hardware.remote_memory_bandwidth_gbps,
            self.source["remote-memory"]["bandwidth-gbps"],
        )
        self.assertEqual(
            self.hardware.remote_memory_type,
            self.source["remote-memory"]["memory-type"],
        )
        self.assertEqual(
            self.hardware.remote_memory_latency_ns,
            self.source["remote-memory"]["latency-ns"],
        )
        self.assertEqual(
            self.hardware.remote_memory_npu_selection,
            self.source["remote-memory"]["npu-selection"],
        )
        self.assertEqual(
            self.hardware.remote_memory_logical_pool,
            self.source["remote-memory"]["logical-pool"],
        )
        self.assertEqual(self.hardware.peak_perf_tflops, self.source["compute"]["peak-perf-tflops"])
        self.assertEqual(self.hardware.metadata["selected-capacity-profile"], self.profile_name)
        self.assertEqual(self.hardware.metadata["selected-capacity-note"], self.profile["note"])

    def test_materializes_runtime_files_from_repo_local_sources(self) -> None:
        columns = self.hardware.mesh_cols
        ranks = [0, 1, columns, columns + 1]
        with tempfile.TemporaryDirectory() as temporary:
            paths = materialize_runtime_configs(
                hardware=self.hardware,
                system_template_path=_SYSTEM_TEMPLATE,
                inference_groups=[("1", ranks)],
                output_dir=Path(temporary),
            )
            system = json.loads(paths.system.read_text(encoding="utf-8"))
            self.assertEqual(system["local-mem-bw"], self.source["local-hbm"]["bandwidth-gbps"])
            self.assertEqual(system["local-mem-latency"], self.source["local-hbm"]["latency-ns"])
            # P11 死键清除回归钉（2026-09-23）：local-mem-capacity-bytes
            # 写入链已退役（C++ 零读者），system.json 不再含该键。
            self.assertNotIn("local-mem-capacity-bytes", system)
            self.assertEqual(system["remote-mem-bw"], self.source["remote-memory"]["bandwidth-gbps"])
            self.assertEqual(system["peak-perf"], self.source["compute"]["peak-perf-tflops"])
            self.assertEqual(system["hbm-kv-restore-bandwidth-sharing"], 1)
            self.assertEqual(system["hbm-bandwidth-contention"], 1)

            remote = json.loads(paths.remote_memory.read_text(encoding="utf-8"))
            self.assertEqual(remote["memory-type"], "PER_NPU_MEMORY_EXPANSION")
            self.assertEqual(remote["remote-mem-bw"], self.source["remote-memory"]["bandwidth-gbps"])
            self.assertEqual(remote["remote-mem-latency"], self.source["remote-memory"]["latency-ns"])
            # P11 死键清除回归钉（2026-09-23）：logical-pool 写入链已退役
            # （C++ 零读者、纯审计透传），remote_memory.json 不再含该键。
            self.assertNotIn("logical-pool", remote)
            self.assertNotIn("npu-selection", remote)
            expected_boundary = [
                row * self.hardware.mesh_cols + column
                for row in range(self.hardware.mesh_rows)
                for column in range(self.hardware.mesh_cols)
                if row in {0, self.hardware.mesh_rows - 1}
                or column in {0, self.hardware.mesh_cols - 1}
            ]
            self.assertEqual(remote["npu-ids"], expected_boundary)

            communicator = json.loads(paths.comm_group.read_text(encoding="utf-8"))
            self.assertEqual(communicator["1"], {"ranks": ranks, "dimensions": [2, 2]})
            network = paths.network.read_text(encoding="utf-8")
            self.assertTrue(network.startswith(f"# Generated from {self.hardware_path}"))
            self.assertIn(
                "npus_count: [ "
                f"{self.hardware.mesh_cols}, {self.hardware.mesh_rows} ]",
                network,
            )
            for path in (paths.system, paths.network, paths.remote_memory, paths.comm_group):
                self.assertTrue(path.read_text(encoding="utf-8").endswith("\n"))

    def test_rejects_duplicate_communicators_and_non_rectangular_groups(self) -> None:
        columns = self.hardware.mesh_cols
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary)
            with self.assertRaisesRegex(ValueError, "Duplicate communicator"):
                materialize_runtime_configs(
                    hardware=self.hardware,
                    system_template_path=_SYSTEM_TEMPLATE,
                    inference_groups=[("1", [0]), ("01", [1])],
                    output_dir=output_dir,
                )
            with self.assertRaisesRegex(ValueError, "rectangle"):
                materialize_runtime_configs(
                    hardware=self.hardware,
                    system_template_path=_SYSTEM_TEMPLATE,
                    inference_groups=[("1", [0, 1, columns])],
                    output_dir=output_dir,
                )

    def test_authored_templates_do_not_duplicate_managed_hardware_fields(self) -> None:
        system_template = json.loads(_SYSTEM_TEMPLATE.read_text(encoding="utf-8"))
        # (2026-10-01 审查修复) 直接 import 守卫清单——手抄副本缺
        # remote-mem-latency 一键（模板混入该键时生产守卫拒、本测试
        # 恒过 = 假覆盖），未来扩键同步失效。
        from config_resolver import _MANAGED_SYSTEM_FIELDS

        self.assertTrue(_MANAGED_SYSTEM_FIELDS.isdisjoint(system_template))
        die_template = json.loads(_DIE_SYSTEM_TEMPLATE.read_text(encoding="utf-8"))
        # die 模板（第一期）同样不得手写任何 managed 硬件字段
        # （die-model-config 由 resolver 注入）。
        self.assertTrue(_MANAGED_SYSTEM_FIELDS.isdisjoint(die_template))
        self.assertFalse((_SH_TEST_DIR / "remote_memory").exists())


def _die_library_available() -> bool:
    try:
        discover_cabi_library()
    except RuntimeError:
        return False
    return True


class DieHardwareTests(unittest.TestCase):
    """die 节硬件源（第一期，契约 §3.1/§3.3/§3.4）：互斥/结构/值域经库
    同源校验、派生标量回流、容量对齐、resolved_die_model.json 物化与
    system 三键注入/冲突门、发现序 fail-closed。"""

    @classmethod
    def setUpClass(cls) -> None:
        if not _die_library_available():
            raise unittest.SkipTest(
                "die cost library (libdie_model_cabi.so) not discoverable; "
                "build the die_model_cabi target first (candidates: env "
                "DIE_MODEL_CABI_PATH -> "
                + " -> ".join(str(p) for p in cabi_library_candidates()) + ")")

    def setUp(self) -> None:
        self.source = json.loads(_HARDWARE_SOURCE.read_text(encoding="utf-8"))

    def test_checked_in_die_source_resolves_derived_scalars(self) -> None:
        hardware = load_hardware_config(_HARDWARE_SOURCE, "validation-die-64g")
        derived = hardware.die_derived
        self.assertIsNotNone(derived)
        self.assertEqual(derived["mesh_n"], 35)
        self.assertEqual(derived["cores"], 1225)
        self.assertAlmostEqual(hardware.peak_perf_tflops, 1492.05, places=9)
        self.assertAlmostEqual(
            hardware.local_hbm_bandwidth_gbps, 1640.0, places=9)
        self.assertEqual(hardware.local_hbm_capacity_bytes, 64_000_000_000)
        self.assertEqual(hardware.die_profile, "mesh_die_v1")
        # 派生带宽与 legacy compute 形态严格连续（0.75/large/800/4x1/512
        # 裁定点；1640 GB/s 与原 local-hbm 配置值逐位相等）。
        self.assertEqual(
            hardware.local_hbm_bandwidth_gbps,
            self.source["local-hbm"]["bandwidth-gbps"])

    def test_die_and_compute_are_mutually_exclusive(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = _legacy_hardware_copy(Path(temporary))
            source = json.loads(path.read_text(encoding="utf-8"))
            source["die"] = json.loads(
                _HARDWARE_SOURCE.read_text(encoding="utf-8"))["die"]
            path.write_text(json.dumps(source, indent=2) + "\n")
            with self.assertRaisesRegex(
                    ValueError, "both 'die' and 'compute.peak-perf-tflops'"):
                load_hardware_config(path, "paper-64gib")

    def test_neither_die_nor_compute_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = _legacy_hardware_copy(Path(temporary))
            source = json.loads(path.read_text(encoding="utf-8"))
            del source["compute"]
            path.write_text(json.dumps(source, indent=2) + "\n")
            with self.assertRaisesRegex(
                    ValueError,
                    "exactly one of 'die' or 'compute.peak-perf-tflops'"):
                load_hardware_config(path, "paper-64gib")

    def test_die_node_key_structure_is_enforced(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bad_die.json"
            die = dict(json.loads(
                _HARDWARE_SOURCE.read_text(encoding="utf-8"))["die"])
            source = json.loads(
                _HARDWARE_SOURCE.read_text(encoding="utf-8"))
            del source["die"]
            source["compute"] = None
            del source["compute"]
            source["die"] = die
            source["die"]["surprise"] = 1
            path.write_text(json.dumps(source, indent=2) + "\n")
            with self.assertRaisesRegex(ValueError, "surprise"):
                load_hardware_config(path, "validation-die-64g")

    def test_illegal_layout_and_core_table_miss_fail_closed(self) -> None:
        base = json.loads(_HARDWARE_SOURCE.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as temporary:
            for mutate, needle in (
                    (lambda d: d.update({"hbm-per-side": 2, "hbm-sides": 2}),
                     "rejected by the die_model cost library"),
                    (lambda d: d.update({"sram-mb-per-core": 1.75}),
                     "rejected by the die_model cost library"),
                    (lambda d: d.update({"noc-width-bits": 300}),
                     "noc-width-bits"),
            ):
                source = json.loads(json.dumps(base))
                mutate(source["die"])
                path = Path(temporary) / "bad.json"
                path.write_text(json.dumps(source, indent=2) + "\n")
                with self.assertRaisesRegex(ValueError, needle):
                    load_hardware_config(path, "validation-die-64g")

    def test_capacity_alignment_reports_both_numbers(self) -> None:
        with self.assertRaisesRegex(
                ValueError,
                r"171798691840.*64000000000|64000000000.*171798691840"):
            load_hardware_config(_HARDWARE_SOURCE, "validation-160gib")

    def test_resolved_die_model_and_system_keys_materialize(self) -> None:
        hardware = load_hardware_config(_HARDWARE_SOURCE, "validation-die-64g")
        with tempfile.TemporaryDirectory() as temporary:
            paths = materialize_runtime_configs(
                hardware=hardware,
                system_template_path=_DIE_SYSTEM_TEMPLATE,
                inference_groups=[("1", [0, 1])],
                output_dir=Path(temporary),
            )
            self.assertIsInstance(paths, RuntimeConfigPaths)
            self.assertIsNotNone(paths.die_model)
            resolved = json.loads(
                paths.die_model.read_text(encoding="utf-8"))
            self.assertEqual(resolved["profile"], "mesh_die_v1")
            self.assertEqual(resolved["die"], hardware.die_node)
            self.assertEqual(
                resolved["derived"]["cores"], 1225)
            # 库摘要原文可 round-trip 重装载（make_hardware 接受 wrapped
            # 形态；此处经 resolver 路径再走一遍）。
            self.assertEqual(
                load_hardware_config(_HARDWARE_SOURCE,
                                     "validation-die-64g").die_summary,
                resolved)
            system = json.loads(paths.system.read_text(encoding="utf-8"))
            self.assertEqual(system["compute-cost-model"], "die")
            self.assertEqual(system["die-model-profile"], "mesh_die_v1")
            self.assertTrue(
                system["die-model-config"].startswith(str(Path(temporary))))
            self.assertNotIn("roofline-enabled", system)
            # §3.4 回流：peak-perf/local-mem-bw 由派生值填充写入。
            self.assertEqual(system["peak-perf"], 1492.05)
            self.assertEqual(system["local-mem-bw"], 1640.0)

    def test_roofline_template_conflicts_with_die_hardware(self) -> None:
        hardware = load_hardware_config(_HARDWARE_SOURCE, "validation-die-64g")
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(
                    ValueError, "roofline-enabled=1"):
                materialize_runtime_configs(
                    hardware=hardware,
                    system_template_path=_SYSTEM_TEMPLATE,
                    inference_groups=[("1", [0, 1])],
                    output_dir=Path(temporary),
                )

    def test_discovery_fail_closed_without_library(self) -> None:
        import os
        from unittest import mock
        # Keep the environment override absent during discovery itself,
        # including when the suite was launched with a freshly built ABI.
        with mock.patch.dict(os.environ):
            os.environ.pop("DIE_MODEL_CABI_PATH", None)
            with mock.patch("config_resolver.cabi_library_candidates",
                            return_value=tuple()):
                with self.assertRaisesRegex(RuntimeError, "die_model_cabi"):
                    discover_cabi_library()


if __name__ == "__main__":
    unittest.main()
