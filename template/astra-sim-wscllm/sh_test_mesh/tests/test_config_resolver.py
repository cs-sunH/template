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
_DIE_PROFILE = "validation-die-64g"
# die 推广前的 legacy 算力值(face_case5_config_c.json 的原 compute 节),
# 供 legacy 回归拷贝使用(checked-in 硬件已切换 die 节)。
_LEGACY_PEAK_PERF_TFLOPS = 261.12


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
    """把 checked-in 硬件换回 legacy 计算描述(删 die/coarse-die 节 + 加 compute 节)。

    阶段2(2026-10-18)出厂默认翻 coarse_die_v2 后 checked-in 硬件带的是
    coarse-die 节;legacy 回归拷贝对两种 die 后端节点都做剥离。"""
    source = json.loads(_HARDWARE_SOURCE.read_text(encoding="utf-8"))
    source.pop("die", None)
    source.pop("coarse-die", None)
    source["compute"] = {"peak-perf-tflops": _LEGACY_PEAK_PERF_TFLOPS}
    path = Path(directory) / "hardware_legacy.json"
    path.write_text(json.dumps(source), encoding="utf-8")
    return path


class ConfigResolverTests(unittest.TestCase):
    """legacy(roofline/compute 节)路径回归;die 节用例在 DieHardwareTests。"""

    def setUp(self) -> None:
        self.source = json.loads(_HARDWARE_SOURCE.read_text(encoding="utf-8"))
        # tests-F11 (2026-10-01): pin the profile by name. The hardware json
        # carries multiple capacity profiles; picking next(iter(...)) (first
        # JSON key) silently re-pointed every assertion at a different
        # profile if the key order ever changed -- the suite would stay green
        # while no longer locking the paper contract.
        self.profile_name = "paper-64gib"
        self.profile = self.source["local-hbm"]["capacity-profiles"][self.profile_name]
        # checked-in 硬件自 die 推广起 die 化;legacy 回归用换回 compute 节
        # 的临时拷贝(与原 checked-in 形态同构)。
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
            self.hardware.remote_memory_type,
            self.source["remote-memory"]["memory-type"],
        )
        # legacy 路径:算力来自 compute.peak-perf-tflops,无 die 派生。
        self.assertEqual(self.hardware.peak_perf_tflops, _LEGACY_PEAK_PERF_TFLOPS)
        self.assertIsNone(self.hardware.die_node)
        self.assertIsNone(self.hardware.die_summary)
        self.assertEqual(self.hardware.die_derived, None)
        self.assertEqual(self.hardware.die_profile, "")
        self.assertEqual(self.hardware.metadata["selected-capacity-profile"], self.profile_name)
        self.assertEqual(self.hardware.metadata["selected-capacity-note"], self.profile["note"])

    def test_rejects_memory_expansion_declarations(self) -> None:
        expanded = json.loads(self.legacy_source.read_text(encoding="utf-8"))
        expanded["remote-memory"]["memory-type"] = "PER_NPU_MEMORY_EXPANSION"
        expanded_path = Path(self._temp.name) / "hardware_expansion.json"
        expanded_path.write_text(json.dumps(expanded), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "NO_MEMORY_EXPANSION"):
            load_hardware_config(expanded_path, self.profile_name)

    def test_materializes_no_expansion_runtime_files_from_repo_local_sources(self) -> None:
        columns = self.hardware.mesh_cols
        ranks = [0, 1, columns, columns + 1]
        output_dir = Path(self._temp.name) / "legacy_runtime"
        paths = materialize_runtime_configs(
            hardware=self.hardware,
            system_template_path=_SYSTEM_TEMPLATE,
            inference_groups=[("1", ranks)],
            output_dir=output_dir,
        )
        system = json.loads(paths.system.read_text(encoding="utf-8"))
        self.assertEqual(system["local-mem-bw"], self.source["local-hbm"]["bandwidth-gbps"])
        self.assertEqual(system["local-mem-latency"], self.source["local-hbm"]["latency-ns"])
        self.assertEqual(system["peak-perf"], _LEGACY_PEAK_PERF_TFLOPS)
        # legacy 路径不注入任何 die 键(managed 注入仅 die 硬件触发)。
        self.assertNotIn("die-model-config", system)
        self.assertNotIn("compute-cost-model", system)
        self.assertIsNone(paths.die_model)
        self.assertFalse((output_dir / "resolved_die_model.json").exists())

        communicator = json.loads(paths.comm_group.read_text(encoding="utf-8"))
        self.assertEqual(communicator["1"], {"ranks": ranks, "dimensions": [2, 2]})
        network = paths.network.read_text(encoding="utf-8")
        self.assertTrue(network.startswith(f"# Generated from {self.legacy_source}"))
        self.assertIn(
            "npus_count: [ "
            f"{self.hardware.mesh_cols}, {self.hardware.mesh_rows} ]",
            network,
        )
        for path in (paths.system, paths.network, paths.comm_group):
            self.assertTrue(path.read_text(encoding="utf-8").endswith("\n"))

    def test_rejects_duplicate_communicators_and_non_rectangular_groups(self) -> None:
        columns = self.hardware.mesh_cols
        output_dir = Path(self._temp.name) / "comm_groups"
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
        # 直接对拍 config_resolver 的源集合：_MANAGED_SYSTEM_FIELDS 扩容时
        # 本断言自动覆盖新键（手抄一份字段清单只会静默漂移）。die 推广后
        # managed 集新增 die-model-config(resolver 注入),两个模板都查。
        for template_path in (_SYSTEM_TEMPLATE, _DIE_SYSTEM_TEMPLATE):
            system_template = json.loads(template_path.read_text(encoding="utf-8"))
            self.assertTrue(_MANAGED_SYSTEM_FIELDS.isdisjoint(system_template))
        self.assertFalse((_SH_TEST_DIR / "remote_memory").exists())

    def test_managed_template_field_rejected_regardless_of_backend(self) -> None:
        template = json.loads(_SYSTEM_TEMPLATE.read_text(encoding="utf-8"))
        template["die-model-config"] = "/nonexistent/resolved_die_model.json"
        template_path = Path(self._temp.name) / "managed_template.json"
        template_path.write_text(json.dumps(template), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "managed hardware field"):
            materialize_runtime_configs(
                hardware=self.hardware,
                system_template_path=template_path,
                inference_groups=[("1", [0, 1])],
                output_dir=Path(self._temp.name) / "managed_runtime",
            )


class DieHardwareTests(unittest.TestCase):
    """die 节解析:互斥/结构/容量对齐/ctypes 派生/物化注入。"""

    def setUp(self) -> None:
        # Explicit validation fixture; the checked-in research hardware stays
        # unmodified and requires the user's mesh-nx/mesh-ny choice.
        # 阶段2(2026-10-18)出厂默认翻 coarse_die_v2 后 checked-in 硬件不再
        # 带 die 节——mesh_die_v1 节点路径回归改用显式内联夹具(23 x 23
        # 显式网格,原 checked-in 授权工程点的 die 参数),其余硬件骨架
        # (mesh/local-hbm/d2d/remote-memory)仍取自 checked-in 源。
        self._grid_temp = tempfile.TemporaryDirectory()
        self.addCleanup(self._grid_temp.cleanup)
        data = json.loads(_HARDWARE_SOURCE.read_text(encoding="utf-8"))
        data.pop("die", None)
        data.pop("coarse-die", None)
        data["die"] = {
            "sram-mb-per-core": 0.75,
            "compute-area-mm2": 800,
            "hbm-per-side": 4,
            "hbm-sides": 1,
            "noc-width-bits": 512,
            "mesh-nx": 23,
            "mesh-ny": 23,
        }
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
        "not found"。若环境里仓内 build 恰好存在(集成构建后),该路径
        可发现是合法态——用候选文件存在性把用例退化为只钉 env-覆盖
        发现序。"""
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
        # 严格连续点(与 face 同裁定):4x410 GB/s = 1640,与旧硬件同值。
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
        # 阶段2(2026-10-18)出厂默认翻 coarse_die_v2 后 checked-in die
        # 模板的 die-model-profile 随之改写;本用例钉的是 mesh_die_v1
        # 节点路径的物化,用显式回写 profile 的临时模板拷贝,与
        # checked-in 默认解耦(新默认的物化由
        # test_checked_in_coarse_die_factory_default + 冒烟链路覆盖)。
        template = json.loads(_DIE_SYSTEM_TEMPLATE.read_text(encoding="utf-8"))
        template["die-model-profile"] = "mesh_die_v1"
        template_path = Path(self._temp.name) / "die_template_v1.json"
        template_path.write_text(json.dumps(template), encoding="utf-8")
        output_dir = Path(self._temp.name) / "die_runtime"
        paths = materialize_runtime_configs(
            hardware=hardware,
            system_template_path=template_path,
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


    def test_checked_in_coarse_die_factory_default(self) -> None:
        """阶段2出厂默认(2026-10-18):checked-in 硬件带完整 coarse-die
        描述符(coarse_die_v2),mesh_die_v1 die 节点退役;容量档
        validation-coarse-144g 与 coarse 库派生容量对齐(fail-closed),
        旧 die 档位在粗粒度后端下被容量对齐拒绝。"""
        original = json.loads(_HARDWARE_SOURCE.read_text(encoding="utf-8"))
        self.assertNotIn("die", original)
        self.assertNotIn("compute", original)
        descriptor = original["coarse-die"]
        self.assertEqual(descriptor["schema"], "coarse_die_v2")
        self.assertEqual(descriptor["architecture"]["hbm_stacks_per_row"], 2)
        hardware = load_hardware_config(_HARDWARE_SOURCE, "validation-coarse-144g")
        self.assertEqual(hardware.die_profile, "coarse_die_v2")
        # 2 行 x 2 栈 x 36 GB = 144 GB;带宽 = 2 行 x 2 栈 x 1.1776 TB/s
        # = 4710.4 GB/s;峰值算力保持类型化速率,不回流通用标量。
        self.assertEqual(hardware.local_hbm_capacity_bytes, 144000000000)
        self.assertAlmostEqual(hardware.local_hbm_bandwidth_gbps, 4710.4)
        self.assertIsNone(hardware.peak_perf_tflops)
        summary = hardware.die_derived["hardware_summary"]
        self.assertEqual(summary["die_core_count"], 117)  # 13 列 x 9 行(2026-10-21 wafer-fit 默认)
        self.assertEqual(int(summary["hbm_capacity_bytes"]), 144000000000)
        self.assertEqual(
            [float(rate) for rate in summary["hbm_row_peak_Bps"]],
            [2355200000000.0, 2355200000000.0])
        with self.assertRaisesRegex(ValueError, "144000000000"):
            load_hardware_config(_HARDWARE_SOURCE, "validation-die-64g")

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
