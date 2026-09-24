"""C18 三硬件孪生 round-trip 测试（C19 入仓重写，2026-09-22）。

原件失于 2026-09-22 /tmp 事故（PROVENANCE §21 / RECOVERY.md），本文件按
C18 交付登记（RECOVERY.md "各已完成卡交付物登记"）与执行计划 §5 C18
验收条款重写；范式沿用 ``tests/test_config_resolver.py``。

覆盖对象（WP7 硬件孪生集，DSE 输入合同）：
- ``hardware/face_case5_config_c_d2d_x05.json``  D2D 2025 GB/s（x0.5）
- ``hardware/face_case5_config_c_d2d_x2.json``   D2D 8100 GB/s（x2）
- ``hardware/face_case5_config_c_d2d_sub.json``  D2D 1200 GB/s（ρ<1 档）

三孪生 × 2 容量档（paper-64gib / validation-160gib）= 6 组合，逐组断言：
1. **exact-key schema 校验过**——``load_hardware_config`` 的
   ``_require_exact_keys`` 全链（顶层/mesh/local-hbm/capacity profile/
   d2d/remote-memory/compute）不 raise 即通过；
2. **ρ 派生与 ``joint.link_quota.derive_rho_eff`` 单位换算后逐组一致**
   ——GB/s → bytes/ns 显式量纲换算（1 GB/s == 1 B/ns 数值恒等，本仓
   ``JointHardwareRates.from_gbps`` 冻结口径）后，孪生 JSON 的
   ``d2d.bandwidth-gbps / local-hbm.bandwidth-gbps`` 与配额模块的
   ``derive_rho_eff`` 逐组同值；孪生 notes 披露的 ρ 近似值（1.2348 /
   4.9390 / 0.7317）与计算值交叉核对（披露↔计算双向钉死）；
3. **canonical 写回 → resolver 重载逐字段零漂移**——源 JSON 经
   ``_serialize_json`` canonical 序列化写临时文件，重载后全部
   ``ResolvedHardware`` 字段（含 metadata 深比较）与首载逐项相等；
4. **对 base 源 diff 严格限于白名单 ``{slug, label,
   d2d.bandwidth-gbps, notes}``**——孪生与 ``face_case5_config_c.json``
   的递归差分路径逐条白名单判定（"孪生只改 D2D 带宽与身份/说明"的
   C18 冻结约束）。
"""

from __future__ import annotations

import json
from pathlib import Path
import re
import sys
import tempfile
import unittest


_SH_TEST_DIR = Path(__file__).resolve().parents[1]
if str(_SH_TEST_DIR) not in sys.path:
    sys.path.insert(0, str(_SH_TEST_DIR))

_WL_DIR = _SH_TEST_DIR / "workload" / "llama2_7b_inference"
if str(_WL_DIR / "joint") not in sys.path:
    sys.path.insert(0, str(_WL_DIR / "joint"))

from config_resolver import (  # noqa: E402
    ResolvedHardware,
    _serialize_json,
    load_hardware_config,
)
from link_quota import derive_q_init, derive_rho_eff  # noqa: E402


_HARDWARE_DIR = _SH_TEST_DIR / "hardware"
_BASE_SOURCE = _HARDWARE_DIR / "face_case5_config_c.json"

# 三孪生（C18 交付冻结）：文件名后缀 → (D2D GB/s, 期望 ρ, 期望 Q_init)。
# Q_init = max(1, floor(ρ))——x05 ρ≈1.2348 → 1（max(1,·) 保底不触），
# x2 ρ≈4.9390 → 4，sub ρ≈0.7317 → 1（max(1,·) 防结构性死锁保底触）。
_TWINS = {
    "d2d_x05": (2025.0, 2025.0 / 1640.0, 1),
    "d2d_x2": (8100.0, 8100.0 / 1640.0, 4),
    "d2d_sub": (1200.0, 1200.0 / 1640.0, 1),
}

#: 孪生对 base 源允许差异的 JSON 路径白名单（C18 冻结：孪生只改 D2D
#: 带宽与身份/说明，其余逐字段 identical）。
_TWIN_DIFF_WHITELIST = {
    ("slug",),
    ("label",),
    ("d2d", "bandwidth-gbps"),
    ("notes",),
}

#: 量纲换算常数（GB/s → bytes/ns；1 GB = 1e9 B、1 s = 1e9 ns）。
_BYTES_PER_GB = 10**9
_NS_PER_SECOND = 10**9


def _gbps_to_bytes_per_ns(gbps: float) -> float:
    """GB/s → bytes/ns 的显式量纲换算。

    数值上与 ``JointHardwareRates.from_gbps`` 的冻结恒等（1 GB/s ==
    1 B/ns）一致；本测试用显式算式而非隐式数值直传，使"单位换算后
    逐组一致"的断言语义自证。
    """
    return gbps * _BYTES_PER_GB / _NS_PER_SECOND


def _recursive_diff_paths(base: object, twin: object, prefix: tuple = ()):
    """递归收集 base 与 twin 的差异路径（叶级；list 按下标对齐）。"""
    if type(base) is not type(twin):
        yield prefix
        return
    if isinstance(base, dict):
        for key in sorted(set(base) | set(twin)):
            child = prefix + (key,)
            if key not in base or key not in twin:
                yield child
            else:
                yield from _recursive_diff_paths(base[key], twin[key], child)
        return
    if isinstance(base, list):
        if len(base) != len(twin):
            yield prefix
            return
        for index, (left, right) in enumerate(zip(base, twin)):
            yield from _recursive_diff_paths(left, right, prefix + (index,))
        return
    if base != twin:
        yield prefix


class HardwareTwinsRoundtripTests(unittest.TestCase):
    """三孪生 × 2 容量档 = 6 组合的 schema/ρ/round-trip/白名单四断言。"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.base_source = json.loads(
            _BASE_SOURCE.read_text(encoding="utf-8"))
        cls.base_hardware = {
            profile: load_hardware_config(_BASE_SOURCE, profile)
            for profile in cls.base_source["local-hbm"][
                "capacity-profiles"]
        }
        cls.twins = {
            suffix: json.loads(
                (_HARDWARE_DIR / f"face_case5_config_c_{suffix}.json")
                .read_text(encoding="utf-8"))
            for suffix in _TWINS
        }

    def _resolved_fields(self, hardware: ResolvedHardware) -> dict:
        """逐字段快照（除 source_path 外的全部 ResolvedHardware 值域；
        source_path 在写回 round-trip 中必然不同，单独断言文件存在）。"""
        return {
            "capacity_profile": hardware.capacity_profile,
            "slug": hardware.slug,
            "label": hardware.label,
            "paper_case": hardware.paper_case,
            "mesh_rows": hardware.mesh_rows,
            "mesh_cols": hardware.mesh_cols,
            "topology_by_network_dimension":
                hardware.topology_by_network_dimension,
            "local_hbm_capacity_bytes": hardware.local_hbm_capacity_bytes,
            "local_hbm_bandwidth_gbps": hardware.local_hbm_bandwidth_gbps,
            "local_hbm_latency_ns": hardware.local_hbm_latency_ns,
            "d2d_bandwidth_gbps": hardware.d2d_bandwidth_gbps,
            "d2d_latency_ns": hardware.d2d_latency_ns,
            "remote_memory_type": hardware.remote_memory_type,
            "remote_memory_bandwidth_gbps":
                hardware.remote_memory_bandwidth_gbps,
            "remote_memory_latency_ns": hardware.remote_memory_latency_ns,
            "remote_memory_npu_selection":
                hardware.remote_memory_npu_selection,
            "remote_memory_logical_pool":
                hardware.remote_memory_logical_pool,
            "peak_perf_tflops": hardware.peak_perf_tflops,
            "metadata": hardware.metadata,
        }

    def test_six_combos_schema_rho_and_roundtrip(self) -> None:
        """6 组合：exact-key schema 过 + ρ 派生一致 + 写回重载零漂移。"""
        profiles = sorted(
            self.base_source["local-hbm"]["capacity-profiles"])
        self.assertEqual(
            len(profiles), 2,
            f"C18 合同 = 三孪生 × 2 容量档，base 源应有且仅有 2 档，"
            f"实得 {profiles}")
        combos = 0
        for suffix, (d2d_gbps, expected_rho, expected_q_init) \
                in _TWINS.items():
            source_path = (
                _HARDWARE_DIR / f"face_case5_config_c_{suffix}.json")
            source = self.twins[suffix]
            for profile in profiles:
                combos += 1
                with self.subTest(twin=suffix, profile=profile):
                    # 1) exact-key schema：load_hardware_config 内
                    #    _require_exact_keys 全链不 raise 即通过。
                    first = load_hardware_config(source_path, profile)
                    self.assertEqual(
                        first.metadata["selected-capacity-profile"],
                        profile)

                    # 2) ρ 派生：孪生 JSON 字段经显式量纲换算后与
                    #    derive_rho_eff 同值；Q_init 派生与冻结期望一致。
                    noc_bns = _gbps_to_bytes_per_ns(
                        source["d2d"]["bandwidth-gbps"])
                    hbm_bns = _gbps_to_bytes_per_ns(
                        source["local-hbm"]["bandwidth-gbps"])
                    self.assertEqual(noc_bns, d2d_gbps * 1.0)
                    self.assertEqual(
                        first.d2d_bandwidth_gbps, d2d_gbps)
                    self.assertEqual(
                        first.local_hbm_bandwidth_gbps,
                        source["local-hbm"]["bandwidth-gbps"])
                    rho = derive_rho_eff(
                        noc_link_bytes_per_ns=noc_bns,
                        local_hbm_bytes_per_ns=hbm_bns)
                    self.assertAlmostEqual(
                        rho, expected_rho, places=12,
                        msg=(f"{suffix}: derive_rho_eff({noc_bns}, "
                             f"{hbm_bns}) 应等于 d2d/hbm 带宽比"))
                    self.assertEqual(derive_q_init(rho), expected_q_init)

                    # 3) canonical 写回 → resolver 重载逐字段零漂移。
                    with tempfile.TemporaryDirectory() as temporary:
                        roundtrip_path = (
                            Path(temporary) / source_path.name)
                        roundtrip_path.write_text(
                            _serialize_json(source), encoding="utf-8")
                        second = load_hardware_config(
                            roundtrip_path, profile)
                        self.assertTrue(second.source_path.exists())
                        self.assertEqual(
                            second.source_path, roundtrip_path)
                    self.assertEqual(
                        self._resolved_fields(first),
                        self._resolved_fields(second),
                        msg=(f"{suffix}/{profile}: canonical 写回重载应"
                             "逐字段零漂移"))
        self.assertEqual(combos, 6)

    def test_twin_rho_disclosure_notes_match_computation(self) -> None:
        """孪生 notes 披露的 ρ 近似值与计算值交叉核对（双向钉死）。"""
        for suffix, (_, expected_rho, _) in _TWINS.items():
            with self.subTest(twin=suffix):
                hbm = self.twins[suffix]["local-hbm"][
                    "bandwidth-gbps"]
                rho = _gbps_to_bytes_per_ns(
                    self.twins[suffix]["d2d"]["bandwidth-gbps"]) \
                    / _gbps_to_bytes_per_ns(hbm)
                rho_note = next(
                    note for note in self.twins[suffix]["notes"]
                    if "Derived rho" in note)
                # 披露注释形如 "Derived rho = B_D2D/B_HBM =
                # 1200.0/1640.0 ~= 0.7317 < 1 ..."——取 "~=" 后的
                # 四位小数近似值。
                match = re.search(r"~=\s*([0-9]+\.[0-9]+)", rho_note)
                self.assertIsNotNone(
                    match, msg=f"{suffix}: notes 未披露 ρ 近似值")
                disclosed = float(match.group(1))
                self.assertAlmostEqual(
                    disclosed, round(expected_rho, 4), places=4,
                    msg=(f"{suffix}: notes 披露 ρ {disclosed} 应与计算值 "
                         f"{expected_rho!r} 的四位小数一致"))
                if suffix == "d2d_sub":
                    self.assertLess(rho, 1.0)
                else:
                    self.assertGreater(rho, 1.0)

    def test_twin_diff_against_base_is_strictly_whitelisted(self) -> None:
        """孪生对 base 的递归 diff 严格限于 {slug, label, d2d 带宽, notes}。"""
        for suffix, (d2d_gbps, _, _) in _TWINS.items():
            with self.subTest(twin=suffix):
                diff_paths = set(_recursive_diff_paths(
                    self.base_source, self.twins[suffix]))
                self.assertTrue(diff_paths, "孪生至少应 differ 于白名单键")
                off_whitelist = diff_paths - _TWIN_DIFF_WHITELIST
                self.assertEqual(
                    off_whitelist, set(),
                    msg=(f"{suffix}: 白名单之外的差异路径 "
                         f"{sorted(off_whitelist)}——违反 C18 孪生冻结"
                         "（只改 D2D 带宽与身份/说明）"))
                # 白名单四键应全部实际发生差异（slug/label/带宽/notes
                # 均换；notes 差异收敛为追加孪生说明行）。
                self.assertEqual(
                    diff_paths, _TWIN_DIFF_WHITELIST,
                    msg=f"{suffix}: 白名单键应全部发生差异 {diff_paths}")
                self.assertEqual(
                    self.twins[suffix]["d2d"]["bandwidth-gbps"], d2d_gbps)
                self.assertEqual(
                    len(self.twins[suffix]["notes"])
                    - len(self.base_source["notes"]), 3)

    def test_twins_resolve_identically_to_base_except_d2d(self) -> None:
        """解析态等价性：同容量档下孪生与 base 的 ResolvedHardware 仅
        slug/label/metadata（身份与 notes）与 d2d 带宽不同，其余全同。"""
        for suffix in _TWINS:
            source_path = (
                _HARDWARE_DIR / f"face_case5_config_c_{suffix}.json")
            for profile, base_hardware in self.base_hardware.items():
                with self.subTest(twin=suffix, profile=profile):
                    twin_hardware = load_hardware_config(
                        source_path, profile)
                    base_fields = self._resolved_fields(base_hardware)
                    twin_fields = self._resolved_fields(twin_hardware)
                    self.assertNotEqual(
                        base_fields["d2d_bandwidth_gbps"],
                        twin_fields["d2d_bandwidth_gbps"])
                    for field in base_fields:
                        if field in ("d2d_bandwidth_gbps", "metadata"):
                            continue
                        if field in ("slug", "label"):
                            self.assertNotEqual(
                                base_fields[field], twin_fields[field])
                            continue
                        self.assertEqual(
                            base_fields[field], twin_fields[field],
                            msg=(f"{suffix}/{profile}: 非白名单字段 "
                                 f"{field} 在孪生与 base 间应相等"))


if __name__ == "__main__":
    unittest.main()
