from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile
import unittest


_SH_TEST_DIR = Path(__file__).resolve().parents[1]
if str(_SH_TEST_DIR) not in sys.path:
    sys.path.insert(0, str(_SH_TEST_DIR))

from config_resolver import (
    _MANAGED_SYSTEM_FIELDS,
    cabi_library_candidates,
    discover_cabi_library,
    load_hardware_config,
    materialize_runtime_configs,
)


_HARDWARE_SOURCE = _SH_TEST_DIR / "hardware" / "face_case5_config_c.json"
_SYSTEM_TEMPLATE = _SH_TEST_DIR / "system" / "llama2_7b_roofline_template.json"
_DIE_SYSTEM_TEMPLATE = _SH_TEST_DIR / "system" / "llama2_7b_die_template.json"
_LEGACY_PEAK_PERF_TFLOPS = 261.12
_DIE_PROFILE = "validation-die-64g"


def _cabi_available() -> bool:
    """die 成本库可发现性(env DIE_MODEL_CABI_PATH -> 仓内 build 路径)。

    不可发现时 die 接受/派生用例跳过(discovery fail-closed 行为由
    test_die_library_discovery_fails_closed 钉住);legacy 用例不依赖库。"""
    try:
        discover_cabi_library()
        return True
    except RuntimeError:
        return False


def _legacy_hardware_path(directory: Path) -> Path:
    """把 checked-in 硬件换回 legacy 计算描述(删 die 节 + 加 compute 节)。

    checked-in 硬件自阶段 C(2026-10-03 face-LRU 随迁)起 die 化;legacy
    回归用例用换回 compute 节的临时拷贝(与原 checked-in 形态同构,
    remote-memory/pool 节保持不变)。"""
    source = json.loads(_HARDWARE_SOURCE.read_text(encoding="utf-8"))
    del source["die"]
    source["compute"] = {"peak-perf-tflops": _LEGACY_PEAK_PERF_TFLOPS}
    path = Path(directory) / "hardware_legacy.json"
    path.write_text(json.dumps(source), encoding="utf-8")
    return path


class ConfigResolverTests(unittest.TestCase):
    def setUp(self) -> None:
        self.source = json.loads(_HARDWARE_SOURCE.read_text(encoding="utf-8"))
        self.profile_name = next(iter(self.source["local-hbm"]["capacity-profiles"]))
        self.profile = self.source["local-hbm"]["capacity-profiles"][self.profile_name]
        # checked-in 硬件已 die 化;legacy 回归用换回 compute 节的临时拷贝。
        self._temp = tempfile.TemporaryDirectory()
        self.addCleanup(self._temp.cleanup)
        self.legacy_source = _legacy_hardware_path(self._temp.name)
        self.hardware = load_hardware_config(self.legacy_source, self.profile_name)

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
        # global-8 对齐 (2026-09-26，原仓已删、本仓 2026-09-28 同步): 死
        # schema 键 "paper-case"（曾必填但无消费者）从规范源删除。
        self.assertNotIn("paper-case", self.source)
        self.assertEqual(
            self.hardware.remote_memory_npu_selection,
            self.source["remote-memory"]["npu-selection"],
        )
        # legacy 路径:算力来自 compute.peak-perf-tflops(临时拷贝),无 die 派生。
        self.assertEqual(self.hardware.peak_perf_tflops, _LEGACY_PEAK_PERF_TFLOPS)
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
            self.assertEqual(system["remote-mem-bw"], self.source["remote-memory"]["bandwidth-gbps"])
            self.assertEqual(system["peak-perf"], _LEGACY_PEAK_PERF_TFLOPS)
            self.assertEqual(system["hbm-kv-restore-bandwidth-sharing"], 1)
            self.assertEqual(system["hbm-bandwidth-contention"], 1)

            remote = json.loads(paths.remote_memory.read_text(encoding="utf-8"))
            self.assertEqual(remote["memory-type"], "PER_NPU_MEMORY_EXPANSION")
            self.assertEqual(remote["remote-mem-bw"], self.source["remote-memory"]["bandwidth-gbps"])
            self.assertEqual(remote["remote-mem-latency"], self.source["remote-memory"]["latency-ns"])
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
            # legacy 拷贝物化:头部注释指向临时拷贝路径(阶段 C 随迁)。
            self.assertTrue(
                network.startswith(f"# Generated from {self.legacy_source}"))
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
            with self.assertRaisesRegex(ValueError, "row-major"):
                materialize_runtime_configs(
                    hardware=self.hardware,
                    system_template_path=_SYSTEM_TEMPLATE,
                    inference_groups=[("1", [0, 1, columns])],
                    output_dir=output_dir,
                )

    def test_asymmetric_single_row_group_dimensions_not_transposed(self) -> None:
        """(2026-10-01) 非对称组钉住 dimensions 顺序 = [列数, 行数]。

        对称组(如 [2,2])无法区分该顺序与写反的 [行数, 列数]——维度序的
        正确性目前仅靠 C++ GeneralComplexTopology 的 rank_map 人工核对
        维持;单行组(dim0=列数=6, dim1=行数=1)让写反的形态变为 [1, 6],
        二者可区分。
        """
        with tempfile.TemporaryDirectory() as temporary:
            paths = materialize_runtime_configs(
                hardware=self.hardware,
                system_template_path=_SYSTEM_TEMPLATE,
                inference_groups=[("1", list(range(self.hardware.mesh_cols)))],
                output_dir=Path(temporary),
            )
            communicator = json.loads(
                paths.comm_group.read_text(encoding="utf-8"))
            self.assertEqual(
                communicator["1"]["dimensions"],
                [self.hardware.mesh_cols, 1])

    def test_authored_templates_do_not_duplicate_managed_hardware_fields(self) -> None:
        system_template = json.loads(_SYSTEM_TEMPLATE.read_text(encoding="utf-8"))
        # 阶段 C 随迁:managed 集以 config_resolver 真值为准(现含
        # die-model-config);两模板均不得自带任何 managed 字段。
        self.assertTrue(_MANAGED_SYSTEM_FIELDS.isdisjoint(system_template))
        die_template = json.loads(_DIE_SYSTEM_TEMPLATE.read_text(encoding="utf-8"))
        self.assertTrue(_MANAGED_SYSTEM_FIELDS.isdisjoint(die_template))
        self.assertFalse((_SH_TEST_DIR / "remote_memory").exists())

class DieHardwareTests(unittest.TestCase):
    """die 节解析(阶段 C):互斥/结构/容量对齐/ctypes 派生/物化注入。"""

    def setUp(self) -> None:
        # Explicit validation fixture; the checked-in research hardware stays
        # unmodified and requires the user's mesh-nx/mesh-ny choice.
        self._grid_temp = tempfile.TemporaryDirectory()
        self.addCleanup(self._grid_temp.cleanup)
        data = json.loads(_HARDWARE_SOURCE.read_text(encoding="utf-8"))
        data["die"].pop("die-width-mm", None)
        data["die"].pop("die-height-mm", None)
        data["die"].update({"mesh-nx": 23, "mesh-ny": 23})
        self.fixture_source = Path(self._grid_temp.name) / "explicit_grid_fixture.json"
        self.fixture_source.write_text(json.dumps(data), encoding="utf-8")
        self.source = json.loads(self.fixture_source.read_text(encoding="utf-8"))
        self.die_node = self.source["die"]
        self._temp = tempfile.TemporaryDirectory()
        self.addCleanup(self._temp.cleanup)
        self.cabi_ready = _cabi_available()

    def _hardware_with(self, mutate) -> Path:
        source = json.loads(self.fixture_source.read_text(encoding="utf-8"))
        mutate(source)
        path = Path(self._temp.name) / "hardware_die.json"
        path.write_text(json.dumps(source), encoding="utf-8")
        return path

    def test_die_and_compute_nodes_are_mutually_exclusive(self) -> None:
        path = self._hardware_with(
            lambda source: source.update(
                {"compute": {"peak-perf-tflops": _LEGACY_PEAK_PERF_TFLOPS}}))
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            load_hardware_config(path, _DIE_PROFILE)

    def test_neither_die_nor_compute_is_rejected(self) -> None:
        def strip(source):
            del source["die"]
        path = self._hardware_with(strip)
        with self.assertRaisesRegex(ValueError, "exactly one of 'die' or"):
            load_hardware_config(path, _DIE_PROFILE)

    def test_die_node_requires_explicit_grid_keys(self) -> None:
        def drop_key(source):
            del source["die"]["noc-width-bits"]
        path = self._hardware_with(drop_key)
        with self.assertRaisesRegex(ValueError, "invalid keys"):
            load_hardware_config(path, _DIE_PROFILE)
        def extra_key(source):
            source["die"]["extra-key"] = 1
        path = self._hardware_with(extra_key)
        with self.assertRaisesRegex(ValueError, "invalid keys"):
            load_hardware_config(path, _DIE_PROFILE)

    def test_die_node_rejects_illegal_noc_width_structurally(self) -> None:
        def bad_width(source):
            source["die"]["noc-width-bits"] = 384
        path = self._hardware_with(bad_width)
        with self.assertRaisesRegex(ValueError, "noc-width-bits"):
            load_hardware_config(path, _DIE_PROFILE)

    def test_die_library_discovery_fails_closed(self) -> None:
        """契约 §7 裁定 1:库不可发现时 die 硬件 fail-closed(不回退)。

        env 清空 + 仓内 build 候选不存在时 discover_cabi_library 必须报
        "not found"。若环境里仓内 build 恰好存在(阶段 E 集成构建后),
        该路径可发现是合法态——用候选文件存在性把用例退化为只钉
        env-覆盖发现序。"""
        previous = os.environ.pop("DIE_MODEL_CABI_PATH", None)
        try:
            if any(path.is_file() for path in cabi_library_candidates()):
                # 仓内 build 已存在:发现序必然成功,env 覆盖仍需可发现。
                env_path = Path(self._temp.name) / "env_lib.so"
                env_path.write_bytes(b"")
                os.environ["DIE_MODEL_CABI_PATH"] = str(env_path)
                self.assertEqual(discover_cabi_library(), env_path)
                return
            with self.assertRaisesRegex(RuntimeError, "not found"):
                discover_cabi_library()
        finally:
            if previous is not None:
                os.environ["DIE_MODEL_CABI_PATH"] = previous
            else:
                os.environ.pop("DIE_MODEL_CABI_PATH", None)

    def test_die_node_rejects_illegal_layout_via_cost_library(self) -> None:
        """(800, 2, 2) 不在合法布局表内(契约 §3.5):库内校验,报错透传。"""
        if not self.cabi_ready:
            self.skipTest("libdie_model_cabi.so not discoverable")
        def illegal_layout(source):
            source["die"]["hbm-per-side"] = 2
            source["die"]["hbm-sides"] = 2
        path = self._hardware_with(illegal_layout)
        with self.assertRaisesRegex(ValueError, "illegal"):
            load_hardware_config(path, _DIE_PROFILE)

    def test_die_node_rejects_unknown_core_table_row_via_cost_library(self) -> None:
        if not self.cabi_ready:
            self.skipTest("libdie_model_cabi.so not discoverable")
        # 3.0 MiB is outside the 32-row Core table:库内校验。
        def bad_sram(source):
            source["die"]["sram-mb-per-core"] = 3.0
        path = self._hardware_with(bad_sram)
        with self.assertRaisesRegex(ValueError, "core table"):
            load_hardware_config(path, _DIE_PROFILE)

    def test_die_capacity_profile_must_match_derived_hbm(self) -> None:
        """契约 §3.4:die 节存在时 profile bytes 必须等于派生 HBM 容量。"""
        if not self.cabi_ready:
            self.skipTest("libdie_model_cabi.so not discoverable")
        with self.assertRaisesRegex(ValueError, "171798691840") as context:
            load_hardware_config(self.fixture_source, "validation-160gib")
        # 错误信息同时给出两边数值(选档与派生容量)。
        self.assertIn("64000000000", str(context.exception))

    def test_die_derived_scalars_flow_into_resolved_hardware(self) -> None:
        """接受路径:派生标量回流(峰值算力/带宽/容量,§3.4 口径)。"""
        if not self.cabi_ready:
            self.skipTest("libdie_model_cabi.so not discoverable")
        hardware = load_hardware_config(self.fixture_source, _DIE_PROFILE)
        derived = hardware.die_derived
        self.assertEqual(derived["mesh_n"], 23)
        self.assertEqual(derived["cores"], 529)
        self.assertEqual(derived["hbm_count"], 4)
        # 严格连续点(契约 §3.5):4x410 GB/s = 1640,与旧硬件同值。
        self.assertEqual(hardware.peak_perf_tflops, derived["pe_peak_tflops"])
        self.assertEqual(hardware.local_hbm_bandwidth_gbps, 1640.0)
        self.assertEqual(
            hardware.local_hbm_capacity_bytes,
            int(float(derived["hbm_capacity_bytes"])))
        self.assertEqual(hardware.die_node, self.die_node)
        self.assertEqual(hardware.die_profile, "mesh_die_v1")

    def test_materializes_die_runtime_files_and_system_injection(self) -> None:
        """resolved_die_model.json 落盘 + system.json 三键注入与冲突门。"""
        if not self.cabi_ready:
            self.skipTest("libdie_model_cabi.so not discoverable")
        hardware = load_hardware_config(self.fixture_source, _DIE_PROFILE)
        output_dir = Path(self._temp.name) / "die_runtime"
        paths = materialize_runtime_configs(
            hardware=hardware,
            system_template_path=_DIE_SYSTEM_TEMPLATE,
            inference_groups=[("1", [0, 1, 6, 7])],
            output_dir=output_dir,
        )
        system = json.loads(paths.system.read_text(encoding="utf-8"))
        self.assertEqual(system["compute-cost-model"], "die")
        self.assertEqual(system["die-model-profile"], "mesh_die_v1")
        die_config = Path(system["die-model-config"])
        self.assertTrue(die_config.is_absolute())
        self.assertEqual(paths.die_model, die_config)
        resolved = json.loads(die_config.read_text(encoding="utf-8"))
        self.assertEqual(resolved["schema_version"], 1)
        self.assertEqual(resolved["profile"], "mesh_die_v1")
        self.assertEqual(resolved["die"], hardware.die_node)
        for key in ("mesh_n", "cores", "pe_peak_tflops", "vec_peak_teq",
                    "sram_usable_bytes_per_core", "link_peak_bytes",
                    "hbm_count", "hbm_capacity_bytes", "hbm_bandwidth_bytes"):
            self.assertIn(key, resolved["derived"])
        # §3.4 回流:system.json 的 peak-perf/local-mem-bw 与库派生一致。
        self.assertEqual(system["peak-perf"], resolved["derived"]["pe_peak_tflops"])
        self.assertEqual(
            system["local-mem-bw"],
            float(resolved["derived"]["hbm_bandwidth_bytes"]) / 1e9)
        self.assertEqual(system["local-mem-latency"], 100)

    def test_die_hardware_with_roofline_template_fails_closed(self) -> None:
        """契约 §3.2 冲突矩阵:die 硬件 + roofline-enabled 模板报错。"""
        if not self.cabi_ready:
            self.skipTest("libdie_model_cabi.so not discoverable")
        hardware = load_hardware_config(self.fixture_source, _DIE_PROFILE)
        with self.assertRaisesRegex(ValueError, "roofline-enabled"):
            materialize_runtime_configs(
                hardware=hardware,
                system_template_path=_SYSTEM_TEMPLATE,
                inference_groups=[("1", [0, 1, 6, 7])],
                output_dir=Path(self._temp.name) / "conflict_runtime",
            )

    def test_die_hardware_with_foreign_cost_model_fails_closed(self) -> None:
        if not self.cabi_ready:
            self.skipTest("libdie_model_cabi.so not discoverable")
        hardware = load_hardware_config(self.fixture_source, _DIE_PROFILE)
        template = json.loads(_DIE_SYSTEM_TEMPLATE.read_text(encoding="utf-8"))
        template.pop("roofline-enabled", None)
        template["compute-cost-model"] = "roofline"
        template_path = Path(self._temp.name) / "foreign_template.json"
        template_path.write_text(json.dumps(template), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "compute-cost-model"):
            materialize_runtime_configs(
                hardware=hardware,
                system_template_path=template_path,
                inference_groups=[("1", [0, 1, 6, 7])],
                output_dir=Path(self._temp.name) / "foreign_runtime",
            )



    def test_checked_in_authorized_engineering_point(self) -> None:
        original = json.loads(_HARDWARE_SOURCE.read_text(encoding="utf-8"))
        self.assertNotIn("core-type", original["die"])
        self.assertEqual((original["die"]["mesh-nx"], original["die"]["mesh-ny"]), (10, 10))
        self.assertEqual((original["die"]["die-width-mm"], original["die"]["die-height-mm"]), (25, 32))
        hardware = load_hardware_config(_HARDWARE_SOURCE, "validation-die-64g")
        self.assertEqual(hardware.die_derived["cores"], 100)
        self.assertAlmostEqual(hardware.die_derived["pe_peak_int4_tops"], 126.6)
        self.assertEqual(hardware.die_derived["die_geometry_mode"], "explicit_width_height")

    def test_optional_dimensions_are_paired_and_physically_checked(self) -> None:
        source = json.loads(self.fixture_source.read_text(encoding="utf-8"))
        for dimensions, needle in (({"die-width-mm": 25}, "provided together"),
                ({"die-width-mm": 25, "die-height-mm": 33}, "must equal"),
                ({"die-width-mm": 10, "die-height-mm": 80}, "independently")):
            current = json.loads(json.dumps(source))
            current["die"].update(dimensions)
            path = Path(self._grid_temp.name) / "invalid_dimensions.json"
            path.write_text(json.dumps(current), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, needle):
                load_hardware_config(path, "validation-die-64g")

    def test_obsolete_core_selector_is_rejected_without_conversion(self) -> None:
        for value in ("small", "large", "core"):
            source = json.loads(self.fixture_source.read_text(encoding="utf-8"))
            source["die"]["core-type"] = value
            path = Path(self._grid_temp.name) / "obsolete_core_selector.json"
            path.write_text(json.dumps(source), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "core-type"):
                load_hardware_config(path, "validation-die-64g")


if __name__ == "__main__":
    unittest.main()
