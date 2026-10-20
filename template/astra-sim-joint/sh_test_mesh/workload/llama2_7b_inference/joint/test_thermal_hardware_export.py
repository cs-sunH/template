"""Hardware-facts export tests (plan V02). Skips when the coarse C ABI is absent.

The resolver's coarse_die_v2 hardware needs ``libcoarse_die_cabi.so``.  When it
is not built (the checked-in default), these tests skip rather than pass: an
environment miss is never reported as a behavioural pass.
"""
import os
import sys
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[4]
_CONFIG_RESOLVER_DIR = _REPO_ROOT / "sh_test_mesh"
_HARDWARE_CONFIG = _CONFIG_RESOLVER_DIR / "hardware" / "face_case5_config_c.json"
_CABI_NAMES = ("libcoarse_die_cabi.so",)

if str(_CONFIG_RESOLVER_DIR) not in sys.path:
    sys.path.insert(0, str(_CONFIG_RESOLVER_DIR))
if str(_REPO_ROOT / "extern" / "thermal_cosim") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "extern" / "thermal_cosim"))


def _cabi_available() -> bool:
    env = os.environ.get("DIE_COARSE_CABI_PATH")
    candidates = [Path(env)] if env else []
    build = _REPO_ROOT / "build" / "astra_analytical" / "build_congestion_aware"
    candidates += [build / "lib" / name for name in _CABI_NAMES]
    candidates += [build / "bin" / name for name in _CABI_NAMES]
    return any(path.is_file() for path in candidates)


@unittest.skipUnless(_cabi_available(), "libcoarse_die_cabi.so not built")
class HardwareExportTests(unittest.TestCase):
    def _facts(self):
        import config_resolver as cr

        hardware = cr.load_hardware_config(_HARDWARE_CONFIG, "validation-coarse-144g")
        return cr.export_thermal_hardware_facts(hardware)

    def test_export_reports_real_hardware_and_missing_facts(self):
        facts = self._facts()
        self.assertEqual(facts["schema_version"], "joint.thermal-hardware-facts.v1")
        self.assertEqual(facts["mesh"]["npus_count"], 54)
        self.assertEqual(len(facts["dies"]), 54)
        self.assertEqual(facts["clock_hz"], 1.0e9)
        self.assertEqual(facts["hbm"]["capacity_bytes_per_die"], 144_000_000_000)
        self.assertFalse(facts["target_calibrated"])
        # Fields the host cannot supply must be reported, not invented.
        self.assertTrue(any("noc" in item for item in facts["missing_host_facts"]))
        self.assertTrue(any("macro" in item for item in facts["missing_host_facts"]))

    def test_target_hardware_does_not_match_the_reference_energy_table(self):
        """Documents the H-section gap: strict energy validation must fail closed."""
        from wafercosim.action_energy import EnergyCalibrationError, EnergyModel

        facts = self._facts()
        with self.assertRaises(EnergyCalibrationError):
            EnergyModel(
                {"calibration": {"energy_table": "configs/energy/research-v1.json",
                                 "strict": True},
                 "thermal": {"initial_k": 300.0}},
                hardware_facts=facts["energy_hardware_facts"])


if __name__ == "__main__":
    unittest.main()
