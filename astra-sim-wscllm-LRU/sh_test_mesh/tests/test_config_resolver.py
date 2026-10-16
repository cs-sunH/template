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
    _MANAGED_SYSTEM_FIELDS,
    cabi_library_candidates,
    discover_cabi_library,
    load_hardware_config,
    materialize_runtime_configs,
)


_HARDWARE_SOURCE = _SH_TEST_DIR / "hardware" / "face_case5_config_c.json"
_SYSTEM_TEMPLATE = _SH_TEST_DIR / "system" / "llama2_7b_roofline_template.json"
_DIE_SYSTEM_TEMPLATE = _SH_TEST_DIR / "system" / "llama2_7b_die_template.json"


def _library_available() -> bool:
    """Die 用例的库可用性门:走真实发现序(env -> 仓内 build 路径)。"""
    try:
        discover_cabi_library()
    except RuntimeError:
        return False
    return True


def _write_legacy_hardware_copy(directory: Path) -> Path:
    """Legacy 路径夹具(阶段 C):checked-in 硬件已切 die 节——legacy
    断言(compute.peak-perf-tflops 派生、容量 = profile 原值等)在
    "换回 compute 节"的临时拷贝上运行,checked-in 断言全保留(DieHardware
    Tests 覆盖 die 节行为)。"""
    source = json.loads(_HARDWARE_SOURCE.read_text(encoding="utf-8"))
    die_node = source.pop("die", None)
    if die_node is None:
        raise AssertionError(
            "checked-in hardware is expected to carry the die node "
            "(stage C); update this fixture shim")
    source["compute"] = {"peak-perf-tflops": 261.12}
    path = directory / "legacy_hardware.json"
    path.write_text(json.dumps(source, indent=2) + "\n", encoding="utf-8")
    return path


class ConfigResolverTests(unittest.TestCase):
    def setUp(self) -> None:
        # 阶段 C:legacy 断言在"换回 compute 节"的临时拷贝上运行
        # (checked-in 硬件现为 die 节;原断言逐条保留)。
        self._temp = tempfile.TemporaryDirectory()
        self.addCleanup(self._temp.cleanup)
        self.hardware_path = _write_legacy_hardware_copy(Path(self._temp.name))
        self.source = json.loads(self.hardware_path.read_text(encoding="utf-8"))
        self.profile_name = "paper-64gib"
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
        self.assertEqual(self.hardware.peak_perf_tflops, self.source["compute"]["peak-perf-tflops"])
        self.assertEqual(self.hardware.metadata["selected-capacity-profile"], self.profile_name)
        self.assertEqual(self.hardware.metadata["selected-capacity-note"], self.profile["note"])

    def test_materializes_runtime_files_from_repo_local_sources(self) -> None:
        columns = self.hardware.mesh_cols
        # M-29 整网覆盖门（2026-09-30）生效后，materialize 的组集必须恰好
        # 覆盖全部物理 NPU；夹具改用与交付 trace_config.csv 同款的
        # 3 行 x 2 列分块铺满整网（组 "1" 即左上角块）。
        group_ranks = [
            [
                row * columns + column
                for row in range(row_block, row_block + 3)
                for column in range(col_block, col_block + 2)
            ]
            for row_block in range(0, self.hardware.mesh_rows, 3)
            for col_block in range(0, columns, 2)
        ]
        inference_groups = [
            (str(index + 1), ranks) for index, ranks in enumerate(group_ranks)
        ]
        ranks = group_ranks[0]
        with tempfile.TemporaryDirectory() as temporary:
            paths = materialize_runtime_configs(
                hardware=self.hardware,
                system_template_path=_SYSTEM_TEMPLATE,
                inference_groups=inference_groups,
                output_dir=Path(temporary),
            )
            system = json.loads(paths.system.read_text(encoding="utf-8"))
            self.assertEqual(system["local-mem-bw"], self.source["local-hbm"]["bandwidth-gbps"])
            self.assertEqual(system["local-mem-latency"], self.source["local-hbm"]["latency-ns"])
            self.assertEqual(system["remote-mem-bw"], self.source["remote-memory"]["bandwidth-gbps"])
            self.assertEqual(system["peak-perf"], self.source["compute"]["peak-perf-tflops"])
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
            self.assertEqual(communicator["1"], {"ranks": ranks, "dimensions": [2, 3]})
            network = paths.network.read_text(encoding="utf-8")
            self.assertTrue(network.startswith(f"# Generated from {self.hardware_path}"))
            self.assertIn(
                "npus_count: [ "
                f"{self.hardware.mesh_cols}, {self.hardware.mesh_rows} ]",
                network,
            )
            for path in (paths.system, paths.network, paths.remote_memory, paths.comm_group):
                self.assertTrue(path.read_text(encoding="utf-8").endswith("\n"))

    def test_impl_value_combination_gate_rejects_ring_all_to_all(self) -> None:
        """值组合门（2026-10-01 移植）：all-to-all 只许 direct/oneDirect 族
        （C++ Ring.cc plain Ring 拒驱动 All_to_All 即 critical+exit(1) 的
        配置侧镜像）；交付模板的 all-to-all-implementation 必须是 direct。
        ring 配置此前在 resolver 侧零诊断、直接漏到 C++ 运行期才爆。"""
        columns = self.hardware.mesh_cols
        group_ranks = [
            [
                row * columns + column
                for row in range(row_block, row_block + 3)
                for column in range(col_block, col_block + 2)
            ]
            for row_block in range(0, self.hardware.mesh_rows, 3)
            for col_block in range(0, columns, 2)
        ]
        inference_groups = [
            (str(index + 1), ranks) for index, ranks in enumerate(group_ranks)
        ]
        with tempfile.TemporaryDirectory() as temporary:
            template_path = Path(temporary) / "template.json"
            template = json.loads(_SYSTEM_TEMPLATE.read_text(encoding="utf-8"))
            template["all-to-all-implementation"] = ["ring", "ring"]
            template_path.write_text(
                json.dumps(template), encoding="utf-8")
            with self.assertRaisesRegex(
                    ValueError, "not a legal combination"):
                materialize_runtime_configs(
                    hardware=self.hardware,
                    system_template_path=template_path,
                    inference_groups=inference_groups,
                    output_dir=Path(temporary),
                )
            # doubleBinaryTree 只许 all-reduce 键。
            template["all-to-all-implementation"] = ["direct", "direct"]
            template["all-gather-implementation"] = ["doubleBinaryTree", "ring"]
            template_path.write_text(
                json.dumps(template), encoding="utf-8")
            with self.assertRaisesRegex(
                    ValueError, "not a legal combination"):
                materialize_runtime_configs(
                    hardware=self.hardware,
                    system_template_path=template_path,
                    inference_groups=inference_groups,
                    output_dir=Path(temporary),
                )
        # 交付模板本身必须过门（all-to-all = direct 族）。
        delivered = json.loads(_SYSTEM_TEMPLATE.read_text(encoding="utf-8"))
        self.assertEqual(
            delivered["all-to-all-implementation"], ["direct", "direct"])

    def test_rejects_duplicate_communicators_and_non_rectangular_groups(self) -> None:
        columns = self.hardware.mesh_cols
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary)
            with self.assertRaisesRegex(ValueError, "Duplicate communicator"):
                materialize_runtime_configs(
                    hardware=self.hardware,
                    system_template_path=_SYSTEM_TEMPLATE,
                    # 同 ID（"01" == 1）重复；两组都用合法 2x2 形状——
                    # 1x1/1xN 组现在在维度下限守卫处更早 fail-closed。
                    inference_groups=[
                        ("1", [0, 1, columns, columns + 1]),
                        ("01", [2, 3, columns + 2, columns + 3]),
                    ],
                    output_dir=output_dir,
                )
            # A4-10(2026-10-01 face 同款移植)后,"contiguous row-major
            # rectangle"集合等价校验被 position-by-position 有序比较蕴含
            # 取代:非矩形集合(L 形)同样以 row-major 序消息 fail-closed。
            with self.assertRaisesRegex(ValueError, "row-major"):
                materialize_runtime_configs(
                    hardware=self.hardware,
                    system_template_path=_SYSTEM_TEMPLATE,
                    inference_groups=[("1", [0, 1, columns])],
                    output_dir=output_dir,
                )
            # A4-10 的直接靶形:集合恰为完整矩形但列表乱序(dim1-fastest)——
            # 旧集合等价校验放行、C++ 按位次消费会静默建转置环;有序比较
            # 在配置期拦截。
            with self.assertRaisesRegex(ValueError, "row-major"):
                materialize_runtime_configs(
                    hardware=self.hardware,
                    system_template_path=_SYSTEM_TEMPLATE,
                    inference_groups=[
                        ("1", [1, 0, columns + 1, columns]),
                    ],
                    output_dir=output_dir,
                )
            # 单行/单列组(1xN / Nx1)以合法 row-major 序通过有序校验,但
            # C++ 侧 RingTopology 要求每个环维度 >= 2 节点——配置期即拦截。
            with self.assertRaisesRegex(ValueError, "2 rows and 2 columns"):
                materialize_runtime_configs(
                    hardware=self.hardware,
                    system_template_path=_SYSTEM_TEMPLATE,
                    inference_groups=[("1", [0, 1])],
                    output_dir=output_dir,
                )

    def test_authored_templates_do_not_duplicate_managed_hardware_fields(self) -> None:
        # 守卫键集直接取自实现(_MANAGED_SYSTEM_FIELDS),避免内联副本与
        # 实现(8 键)漂移导致单元验证弱于运行时守卫。
        system_template = json.loads(_SYSTEM_TEMPLATE.read_text(encoding="utf-8"))
        self.assertTrue(_MANAGED_SYSTEM_FIELDS.isdisjoint(system_template))
        self.assertFalse((_SH_TEST_DIR / "remote_memory").exists())


@unittest.skipUnless(
    _library_available(),
    "libdie_model_cabi.so not discoverable (env DIE_MODEL_CABI_PATH -> "
    + " -> ".join(str(path) for path in cabi_library_candidates())
    + "); build the die_model_cabi target first",
)
class DieHardwareTests(unittest.TestCase):
    """阶段 C:die 节硬件源(契约 §3.1/§3.3/§3.4/§7 裁定 1)。"""

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
        self.die = self.source["die"]

    def _copy_with_die(self, directory: Path, die: dict) -> Path:
        source = json.loads(self.fixture_source.read_text(encoding="utf-8"))
        source.pop("die", None)
        source["die"] = die
        path = directory / "hw_die.json"
        path.write_text(json.dumps(source, indent=2) + "\n", encoding="utf-8")
        return path

    def test_explicit_fixture_preserves_user_hardware_parameters(self) -> None:
        """Explicit test input: 23x23 / 0.75 MiB / 800 mm2 / HBM 4x1 / 512 bit
        (带宽 1640 GB/s 与旧硬件严格连续;容量派生 64 GB)。"""
        self.assertEqual(self.source.get("compute"), None)
        self.assertEqual(self.die, {
            "sram-mb-per-core": 0.75,
            "mesh-nx": 23, "mesh-ny": 23,
            "compute-area-mm2": 800,
            "hbm-per-side": 4,
            "hbm-sides": 1,
            "noc-width-bits": 512,
        })

    def test_derived_scalars_backflow(self) -> None:
        hardware = load_hardware_config(self.fixture_source, "validation-die-64g")
        derived = hardware.die_derived
        self.assertEqual(derived["mesh_n"], 23)
        self.assertEqual(derived["cores"], 529)
        self.assertAlmostEqual(derived["pe_peak_tflops"], 669.714, places=9)
        self.assertAlmostEqual(
            derived["hbm_bandwidth_bytes"] / 1e9, 1640.0, places=9)
        self.assertEqual(derived["hbm_capacity_bytes"], 64000000000)
        self.assertEqual(hardware.peak_perf_tflops, derived["pe_peak_tflops"])
        self.assertAlmostEqual(
            hardware.local_hbm_bandwidth_gbps,
            derived["hbm_bandwidth_bytes"] / 1e9,
            places=9,
        )
        self.assertEqual(
            hardware.local_hbm_capacity_bytes,
            derived["hbm_capacity_bytes"])
        self.assertEqual(hardware.die_profile, "mesh_die_v1")

    def test_die_and_compute_are_mutually_exclusive(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = self._copy_with_die(Path(temporary), dict(self.die))
            data = json.loads(path.read_text(encoding="utf-8"))
            data["compute"] = {"peak-perf-tflops": 261.12}
            path.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(
                    ValueError, "must not declare both"):
                load_hardware_config(path, "validation-die-64g")

    def test_missing_compute_description_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = self._copy_with_die(Path(temporary), dict(self.die))
            data = json.loads(path.read_text(encoding="utf-8"))
            data.pop("die")
            path.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(
                    ValueError, "exactly one of 'die' or"):
                load_hardware_config(path, "paper-64gib")

    def test_die_node_explicit_grid_structure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            for mutation in (
                {"sram-mb-per-core": 0.75},  # 缺五键
                {"unknown-key": 1},
            ):
                path = self._copy_with_die(Path(temporary), mutation)
                with self.assertRaisesRegex(ValueError, "invalid keys"):
                    load_hardware_config(path, "validation-die-64g")
            # 位宽值域
            for width in (100, 2048):
                die = dict(self.die)
                die["noc-width-bits"] = width
                path = self._copy_with_die(Path(temporary), die)
                with self.assertRaisesRegex(
                        ValueError, "noc-width-bits must be one of"):
                    load_hardware_config(path, "validation-die-64g")

    def test_illegal_layout_and_core_table_miss_rejected_by_library(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            die = dict(self.die)
            die["hbm-per-side"], die["hbm-sides"] = 2, 2
            path = self._copy_with_die(Path(temporary), die)
            with self.assertRaisesRegex(ValueError, "rejected by the die_model"):
                load_hardware_config(path, "validation-die-64g")
            die = dict(self.die)
            die["sram-mb-per-core"] = 3.0  # 核表未命中
            path = self._copy_with_die(Path(temporary), die)
            with self.assertRaisesRegex(ValueError, "rejected by the die_model"):
                load_hardware_config(path, "validation-die-64g")

    def test_capacity_alignment_reports_both_values(self) -> None:
        # checked-in legacy profile 在 die 节下被拒,报错含两侧数值。
        with self.assertRaisesRegex(
                ValueError,
                r"171798691840.*64000000000|64000000000.*171798691840"):
            load_hardware_config(self.fixture_source, "validation-160gib")

    def test_resolved_die_model_and_system_injection(self) -> None:
        hardware = load_hardware_config(self.fixture_source, "validation-die-64g")
        columns = hardware.mesh_cols
        group_ranks = [
            [
                row * columns + column
                for row in range(row_block, row_block + 3)
                for column in range(col_block, col_block + 2)
            ]
            for row_block in range(0, hardware.mesh_rows, 3)
            for col_block in range(0, columns, 2)
        ]
        inference_groups = [
            (str(index + 1), ranks) for index, ranks in enumerate(group_ranks)
        ]
        with tempfile.TemporaryDirectory() as temporary:
            paths = materialize_runtime_configs(
                hardware=hardware,
                system_template_path=_DIE_SYSTEM_TEMPLATE,
                inference_groups=inference_groups,
                output_dir=Path(temporary),
            )
            self.assertIsNotNone(paths.die_model)
            resolved = json.loads(
                paths.die_model.read_text(encoding="utf-8"))
            # 库 §3.3 摘要原文可 round-trip(再装载同一 die 节)。
            reloaded = load_hardware_config(
                self.fixture_source, "validation-die-64g")
            self.assertEqual(
                resolved["derived"]["hbm_capacity_bytes"],
                reloaded.die_derived["hbm_capacity_bytes"])
            system = json.loads(paths.system.read_text(encoding="utf-8"))
            self.assertEqual(system["compute-cost-model"], "die")
            self.assertEqual(system["die-model-profile"], "mesh_die_v1")
            self.assertTrue(
                Path(system["die-model-config"]).is_absolute())
            self.assertEqual(
                Path(system["die-model-config"]),
                paths.die_model.resolve())
            # 派生回流(§3.4):peak-perf/local-mem-bw = 库派生值。
            self.assertEqual(system["peak-perf"], hardware.peak_perf_tflops)
            self.assertEqual(
                system["local-mem-bw"], hardware.local_hbm_bandwidth_gbps)

    def test_die_conflicts_with_roofline_template(self) -> None:
        hardware = load_hardware_config(self.fixture_source, "validation-die-64g")
        columns = hardware.mesh_cols
        group_ranks = [
            [
                row * columns + column
                for row in range(row_block, row_block + 3)
                for column in range(col_block, col_block + 2)
            ]
            for row_block in range(0, hardware.mesh_rows, 3)
            for col_block in range(0, columns, 2)
        ]
        inference_groups = [
            (str(index + 1), ranks) for index, ranks in enumerate(group_ranks)
        ]
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(
                    ValueError, "roofline-enabled=1"):
                materialize_runtime_configs(
                    hardware=hardware,
                    system_template_path=_SYSTEM_TEMPLATE,
                    inference_groups=inference_groups,
                    output_dir=Path(temporary),
                )

    def test_die_model_config_is_managed(self) -> None:
        die_template = json.loads(
            _DIE_SYSTEM_TEMPLATE.read_text(encoding="utf-8"))
        self.assertNotIn("die-model-config", die_template)
        self.assertIn("die-model-config", _MANAGED_SYSTEM_FIELDS)


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
