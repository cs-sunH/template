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
_COARSE_PROFILE = "validation-coarse-144g"
# 阶段 C 前的 legacy 算力值(face_case5_config_c.json 的原 compute 节),
# 供 legacy 回归拷贝使用(checked-in 硬件已切换 die 节)。
_LEGACY_PEAK_PERF_TFLOPS = 261.12
# 阶段 2(2026-10-18)出厂默认翻 coarse_die_v2:checked-in 硬件顶层改为
# coarse-die 完整描述符,不再携带 die 节。die 节形态的回归(互斥/结构/
# 容量对齐/物化注入)改由本授权工程点字面量构造的 fixture 承载——即
# newdie 2026-10-16 工程点(10x10 核阵 / 25x32 mm / 800 mm2),断言口径
# 与 checked-in die 时代逐项一致。
_DIE_POINT_NODE = {
    "sram-mb-per-core": 0.75,
    "mesh-nx": 10,
    "mesh-ny": 10,
    "die-width-mm": 25,
    "die-height-mm": 32,
    "compute-area-mm2": 800,
    "hbm-per-side": 4,
    "hbm-sides": 1,
    "noc-width-bits": 512,
}


def _coarse_cabi_available() -> bool:
    """libcoarse_die_cabi.so 可发现性(env DIE_COARSE_CABI_PATH -> 仓内
    build 路径,与 config_resolver.derive_coarse_die_hardware 同一发现序)。

    不可发现时 coarse 用例跳过(库缺位的 fail-closed 行为由
    derive_coarse_die_hardware 的 ValueError 承担,此处只做裸仓守卫)。"""
    candidates = []
    env_value = os.environ.get("DIE_COARSE_CABI_PATH")
    if env_value:
        candidates.append(Path(env_value))
    build = _SH_TEST_DIR.parent / "build" / "astra_analytical" / "build_congestion_aware"
    candidates.extend((build / "lib" / "libcoarse_die_cabi.so",
                       build / "bin" / "libcoarse_die_cabi.so"))
    return any(path.is_file() for path in candidates)


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
    """把 checked-in 硬件换回 legacy 计算描述(删 die/coarse-die 节 +
    加 compute 节;阶段 2 起 checked-in 默认是 coarse-die)。"""
    source = json.loads(_HARDWARE_SOURCE.read_text(encoding="utf-8"))
    source.pop("die", None)
    source.pop("coarse-die", None)
    source["compute"] = {"peak-perf-tflops": _LEGACY_PEAK_PERF_TFLOPS}
    path = Path(directory) / "hardware_legacy.json"
    path.write_text(json.dumps(source), encoding="utf-8")
    return path


class ConfigResolverTests(unittest.TestCase):
    """legacy(roofline/compute 节)路径回归 + die 节接受/拒绝/物化。"""

    def setUp(self) -> None:
        self.source = json.loads(_HARDWARE_SOURCE.read_text(encoding="utf-8"))
        self.profile_name = next(iter(self.source["local-hbm"]["capacity-profiles"]))
        self.profile = self.source["local-hbm"]["capacity-profiles"][self.profile_name]
        # checked-in 硬件自阶段 C 起 die 化;legacy 回归用换回 compute 节
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
        # legacy 路径:算力来自 compute.peak-perf-tflops,无 die 派生。
        self.assertEqual(
            self.hardware.peak_perf_tflops,
            _LEGACY_PEAK_PERF_TFLOPS,
        )
        self.assertIsNone(self.hardware.die_node)
        self.assertIsNone(self.hardware.die_summary)
        self.assertEqual(self.hardware.die_derived, None)
        self.assertEqual(self.hardware.die_profile, "")
        # 2026-09-28: remote_memory_type is no longer stored on
        # ResolvedHardware (no downstream consumer); the NO_MEMORY_EXPANSION
        # fail-closed check at parse time is pinned by
        # test_rejects_memory_expansion_declarations below.
        # A4-13/global-7/global-8 (2026-09-26): the dead schema keys are
        # deleted from the canonical source -- remote-memory
        # "bandwidth-gbps" (backend removed, value never read) and
        # "paper-case" (required but unconsumed).
        self.assertNotIn("bandwidth-gbps", self.source["remote-memory"])
        self.assertNotIn("paper-case", self.source)

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
        paths = materialize_runtime_configs(
            hardware=self.hardware,
            system_template_path=_SYSTEM_TEMPLATE,
            inference_groups=[("1", ranks)],
            output_dir=Path(self._temp.name) / "legacy_runtime",
        )
        system = json.loads(paths.system.read_text(encoding="utf-8"))
        self.assertEqual(system["local-mem-bw"], self.source["local-hbm"]["bandwidth-gbps"])
        self.assertEqual(system["local-mem-latency"], self.source["local-hbm"]["latency-ns"])
        self.assertEqual(system["peak-perf"], _LEGACY_PEAK_PERF_TFLOPS)
        # legacy 路径不注入任何 die 键(managed 注入仅 die 硬件触发)。
        self.assertNotIn("die-model-config", system)
        self.assertNotIn("compute-cost-model", system)
        self.assertIsNone(paths.die_model)
        self.assertFalse(
            (Path(self._temp.name) / "legacy_runtime" / "resolved_die_model.json").exists())

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
        # A4-10 (2026-09-26): the separate rectangle set check is deleted as
        # subsumed; the ordered (dim0-fastest row-major) comparison is the
        # single fail-closed check and rejects this non-rectangular group
        # too, so the regex now pins that message.
        with self.assertRaisesRegex(ValueError, "row-major"):
            materialize_runtime_configs(
                hardware=self.hardware,
                system_template_path=_SYSTEM_TEMPLATE,
                inference_groups=[("1", [0, 1, columns])],
                output_dir=output_dir,
            )
        # A same-set rectangle listed out of order must still be rejected by
        # the surviving ordered comparison.
        with self.assertRaisesRegex(ValueError, "row-major"):
            materialize_runtime_configs(
                hardware=self.hardware,
                system_template_path=_SYSTEM_TEMPLATE,
                inference_groups=[("1", [1, 0, columns + 1, columns])],
                output_dir=output_dir,
            )

    def test_authored_templates_do_not_duplicate_managed_hardware_fields(self) -> None:
        for template in (_SYSTEM_TEMPLATE, _DIE_SYSTEM_TEMPLATE):
            system_template = json.loads(template.read_text(encoding="utf-8"))
            # 2026-09-28: local-mem-capacity-bytes/note deleted from the
            # managed set (no writer anywhere in the repository since A.2
            # removed the remote memory backend). 2026-10-01: assert against
            # the live config_resolver._MANAGED_SYSTEM_FIELDS instead of a
            # local literal -- the literal had drifted (it still listed the
            # since-deleted "peak-perf-note") and this test's job is to
            # mirror the resolver's current managed set, not to snapshot it.
            # 2026-10-03 (stage C): the managed set additionally covers the
            # resolver-injected die-model-config absolute path.
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
    """die 节解析(阶段 C):互斥/结构/容量对齐/ctypes 派生/物化注入。"""

    def setUp(self) -> None:
        # Explicit validation fixture; the checked-in research hardware stays
        # unmodified and requires the user's mesh-nx/mesh-ny choice.
        # 阶段 2 起 checked-in 顶层是 coarse-die;die 节 fixture 从授权
        # 工程点字面量重建(值与 checked-in die 时代的节点逐键一致)。
        self._grid_temp = tempfile.TemporaryDirectory()
        self.addCleanup(self._grid_temp.cleanup)
        data = json.loads(_HARDWARE_SOURCE.read_text(encoding="utf-8"))
        data.pop("coarse-die", None)
        data["die"] = json.loads(json.dumps(_DIE_POINT_NODE))
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
        # 阶段 2 适配:出厂 die 模板已翻 coarse_die_v2,die(mesh_die_v1)
        # 形态的物化回归改用显式 mesh_die_v1 模板拷贝(键集与出厂模板
        # 相同,仅 profile 回 die 形态值;断言口径不变)。
        template = json.loads(_DIE_SYSTEM_TEMPLATE.read_text(encoding="utf-8"))
        template["die-model-profile"] = "mesh_die_v1"
        template_path = Path(self._temp.name) / "die_v1_template.json"
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


    def test_checked_in_authorized_engineering_point(self) -> None:
        # 阶段 2 适配:checked-in 顶层已翻 coarse-die,die 工程点的钉子
        # 移到 _DIE_POINT_NODE fixture 上(值与原 checked-in die 节逐键
        # 相同,断言原样保留)。
        original = json.loads(_HARDWARE_SOURCE.read_text(encoding="utf-8"))
        self.assertNotIn("core-type", _DIE_POINT_NODE)
        self.assertNotIn("die", original)
        self.assertNotIn("compute", original)
        self.assertEqual(
            (_DIE_POINT_NODE["mesh-nx"], _DIE_POINT_NODE["mesh-ny"]), (10, 10))
        self.assertEqual(
            (_DIE_POINT_NODE["die-width-mm"], _DIE_POINT_NODE["die-height-mm"]),
            (25, 32))
        fixture = json.loads(_HARDWARE_SOURCE.read_text(encoding="utf-8"))
        fixture.pop("coarse-die", None)
        fixture["die"] = json.loads(json.dumps(_DIE_POINT_NODE))
        fixture_path = Path(self._temp.name) / "die_point.json"
        fixture_path.write_text(json.dumps(fixture), encoding="utf-8")
        hardware = load_hardware_config(fixture_path, "validation-die-64g")
        self.assertEqual(hardware.die_derived["cores"], 100)
        self.assertAlmostEqual(hardware.die_derived["pe_peak_int4_tops"], 126.6)
        self.assertEqual(hardware.die_derived["die_geometry_mode"], "explicit_width_height")

    def test_checked_in_coarse_die_v2_factory_default(self) -> None:
        """阶段 2 出厂默认:checked-in 硬件 = coarse-die 完整描述符。

        钉住描述符身份(schema/源修订)、三互斥(die/compute 均不在)、
        派生标量(容量 144 GB = 2 行 x 2 栈 x 36 GB、带宽 4710.4 GB/s、
        无 legacy peak 标量)与 die_profile 口径。"""
        if not _coarse_cabi_available():
            self.skipTest("libcoarse_die_cabi.so not discoverable")
        original = json.loads(_HARDWARE_SOURCE.read_text(encoding="utf-8"))
        coarse_node = original["coarse-die"]
        self.assertEqual(coarse_node["schema"], "coarse_die_v2")
        self.assertEqual(
            coarse_node["source_revision"],
            "41f119135fd2d4cc87292f59b4b00adc918eece0")
        hardware = load_hardware_config(_HARDWARE_SOURCE, _COARSE_PROFILE)
        self.assertEqual(hardware.die_profile, "coarse_die_v2")
        self.assertIsNone(hardware.die_node)
        self.assertEqual(hardware.coarse_die_config, coarse_node)
        self.assertEqual(hardware.local_hbm_capacity_bytes, 144_000_000_000)
        self.assertIsNone(hardware.peak_perf_tflops)
        self.assertAlmostEqual(hardware.local_hbm_bandwidth_gbps, 4710.4)
        summary = hardware.die_derived["hardware_summary"]
        self.assertEqual(summary["die_core_count"], 88)
        self.assertEqual(summary["peak_int4_ops_per_s"], 7_040_000_000_000.0)

    def test_coarse_die_capacity_profile_must_match_derived_hbm(self) -> None:
        """契约 §3.4 的 coarse 口径:profile bytes 必须等于 coarse 派生
        per-die HBM 容量(144 GB);错档报错信息给出两边数值。"""
        if not _coarse_cabi_available():
            self.skipTest("libcoarse_die_cabi.so not discoverable")
        with self.assertRaisesRegex(ValueError, "171798691840") as context:
            load_hardware_config(_HARDWARE_SOURCE, "validation-160gib")
        self.assertIn("144000000000", str(context.exception))

    def test_coarse_die_node_is_mutually_exclusive_with_die_and_compute(self) -> None:
        source = json.loads(_HARDWARE_SOURCE.read_text(encoding="utf-8"))
        for extra in ({"die": dict(_DIE_POINT_NODE)},
                      {"compute": {"peak-perf-tflops": _LEGACY_PEAK_PERF_TFLOPS}}):
            current = json.loads(json.dumps(source))
            current.update(extra)
            path = Path(self._temp.name) / "coarse_exclusive.json"
            path.write_text(json.dumps(current), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "mutually exclusive"):
                load_hardware_config(path, _COARSE_PROFILE)

    def test_materializes_coarse_die_runtime_files_and_system_injection(self) -> None:
        """coarse 物化:resolved_die_model.json = 描述符原文(schema
        coarse_die_v2),system.json 注入 die-model-config 绝对路径、
        带宽回流 4710.4 GB/s、无 legacy peak-perf。"""
        if not _coarse_cabi_available():
            self.skipTest("libcoarse_die_cabi.so not discoverable")
        hardware = load_hardware_config(_HARDWARE_SOURCE, _COARSE_PROFILE)
        output_dir = Path(self._temp.name) / "coarse_runtime"
        paths = materialize_runtime_configs(
            hardware=hardware,
            system_template_path=_DIE_SYSTEM_TEMPLATE,
            inference_groups=[("1", [0, 1, 6, 7])],
            output_dir=output_dir,
        )
        system = json.loads(paths.system.read_text(encoding="utf-8"))
        self.assertEqual(system["compute-cost-model"], "die")
        self.assertEqual(system["die-model-profile"], "coarse_die_v2")
        self.assertNotIn("roofline-enabled", system)
        die_config = Path(system["die-model-config"])
        self.assertTrue(die_config.is_absolute())
        self.assertEqual(paths.die_model, die_config)
        resolved = json.loads(die_config.read_text(encoding="utf-8"))
        self.assertEqual(resolved["schema"], "coarse_die_v2")
        self.assertEqual(resolved, hardware.coarse_die_config)
        self.assertNotIn("peak-perf", system)
        self.assertEqual(system["local-mem-bw"], 4710.4)
        self.assertEqual(system["local-mem-latency"], 100)

    def test_coarse_die_with_wrong_template_profile_fails_closed(self) -> None:
        """system 模板 profile 不是 coarse_die_v2 时物化 fail-closed。"""
        if not _coarse_cabi_available():
            self.skipTest("libcoarse_die_cabi.so not discoverable")
        hardware = load_hardware_config(_HARDWARE_SOURCE, _COARSE_PROFILE)
        template = json.loads(_DIE_SYSTEM_TEMPLATE.read_text(encoding="utf-8"))
        template["die-model-profile"] = "mesh_die_v1"
        template_path = Path(self._temp.name) / "wrong_profile_template.json"
        template_path.write_text(json.dumps(template), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "coarse_die_v2"):
            materialize_runtime_configs(
                hardware=hardware,
                system_template_path=template_path,
                inference_groups=[("1", [0, 1, 6, 7])],
                output_dir=Path(self._temp.name) / "wrong_profile_runtime",
            )

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
