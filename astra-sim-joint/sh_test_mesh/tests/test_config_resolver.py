from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest


_SH_TEST_DIR = Path(__file__).resolve().parents[1]
if str(_SH_TEST_DIR) not in sys.path:
    sys.path.insert(0, str(_SH_TEST_DIR))

from config_resolver import load_hardware_config, materialize_runtime_configs


_HARDWARE_SOURCE = _SH_TEST_DIR / "hardware" / "face_case5_config_c.json"
_SYSTEM_TEMPLATE = _SH_TEST_DIR / "system" / "llama2_7b_roofline_template.json"


class ConfigResolverTests(unittest.TestCase):
    def setUp(self) -> None:
        self.source = json.loads(_HARDWARE_SOURCE.read_text(encoding="utf-8"))
        self.profile_name = next(iter(self.source["local-hbm"]["capacity-profiles"]))
        self.profile = self.source["local-hbm"]["capacity-profiles"][self.profile_name]
        self.hardware = load_hardware_config(_HARDWARE_SOURCE, self.profile_name)

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
            self.assertTrue(network.startswith(f"# Generated from {_HARDWARE_SOURCE}"))
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
        managed_fields = {
            "peak-perf",
            "peak-perf-note",
            "local-mem-bw",
            "local-mem-latency",
            "local-mem-capacity-bytes",
            "local-mem-capacity-note",
            "remote-mem-bw",
        }
        self.assertTrue(managed_fields.isdisjoint(system_template))
        self.assertFalse((_SH_TEST_DIR / "remote_memory").exists())


if __name__ == "__main__":
    unittest.main()
