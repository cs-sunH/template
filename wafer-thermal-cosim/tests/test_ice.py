"""Integration checks for the persistent 3D-ICE bridge.

Run with ``python3 -m unittest tests.test_ice -v`` after building
``build/ice_bridge``. The test skips when the optional native tool is absent.
"""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from wafercosim.ice import ThreeDiceThermal
from wafercosim.thermal import LiquidRC


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/thermal/3dice_8die_reference.json"
BRIDGE = ROOT / "build/ice_bridge"


class ThreeDiceGeometryTests(unittest.TestCase):
    """Geometry and coolant conversion need no optional native executable."""

    def test_reference_domain_covers_explicit_thermal_geometry(self):
        config = json.loads(CONFIG.read_text())
        backend = ThreeDiceThermal.__new__(ThreeDiceThermal)
        backend.config = config
        backend.blocks = LiquidRC(config).blocks
        backend.ice_block_rects = backend._rotated_block_rects()
        options = config["thermal"]["three_dice"]
        for name in ("first_wall_length_um", "last_wall_length_um", "channel_length_um",
                     "wall_length_um", "grid_cell_width_um"):
            setattr(backend, name, options[name])
        length, width, channels = backend._domain_geometry()
        self.assertGreaterEqual(length, config["geometry"]["die_rows"]
                                * config["geometry"]["module_height_m"] * 1e6)
        self.assertGreaterEqual(width, config["geometry"]["die_columns"]
                               * config["geometry"]["module_width_m"] * 1e6)
        for x, y, w, h in backend.ice_block_rects:
            self.assertGreaterEqual(length, x + w)
            self.assertGreaterEqual(width, y + h)
        self.assertGreater(channels, 0)

    def test_flow_uses_thermal_die_rows_without_host_network_dimensions(self):
        config = json.loads(CONFIG.read_text())
        config["geometry"]["die_rows"] = 3
        config["thermal"]["coolant_mass_flow_kg_s_by_row"] = [0.001] * 3
        expected = 0.003 / config["thermal"]["coolant_density_kg_m3"] * 60e6
        self.assertAlmostEqual(ThreeDiceThermal._flow_ml_min_from_rc_config(config), expected)
        config["thermal"]["coolant_mass_flow_kg_s_by_row"] = [0.001, 0.002, 0.001]
        with self.assertRaisesRegex(ValueError, "nonuniform"):
            ThreeDiceThermal._flow_ml_min_from_rc_config(config)


@unittest.skipUnless(BRIDGE.is_file(), "build the optional 3D-ICE bridge first")
class ThreeDiceThermalTests(unittest.TestCase):
    def test_persistent_state_zero_power_and_conservative_block_mapping(self):
        config = json.loads(CONFIG.read_text())
        with tempfile.TemporaryDirectory(prefix="wafercosim-3dice-test-") as tmp:
            thermal = ThreeDiceThermal(config, BRIDGE, tmp)
            try:
                self.assertEqual(thermal.nsolid, 48)
                self.assertEqual(thermal.ndie, 8)
                self.assertAlmostEqual(thermal.flow_ml_min, 384.0, places=9)
                self.assertIsNone(thermal.energy_residual_j)
                with self.assertRaisesRegex(ValueError, "fixed step"):
                    thermal.step(np.zeros(thermal.nsolid), 0.005)

                zero = thermal.step(np.zeros(thermal.nsolid), 0.01)
                np.testing.assert_allclose(zero, np.full(thermal.nsolid, 300.0), atol=1e-3, rtol=0)

                power = np.zeros(thermal.nsolid)
                for die in range(thermal.ndie):
                    for index, block in enumerate(thermal.die_blocks[die]["compute"]):
                        power[block] = 0.25 + 0.025 * die + 0.01 * index
                    power[thermal.die_blocks[die]["hbm"][0]] = 0.1
                    power[thermal.die_blocks[die]["router"][0]] = 0.05
                heated = thermal.step(power, 0.01)

                self.assertEqual(thermal.native_step_count, 2)
                self.assertAlmostEqual(thermal.time_s, 0.02, places=12)
                self.assertTrue(np.all(np.isfinite(heated)))
                self.assertGreater(float(heated.max()), float(zero.max()))
                expected_die_max = np.array([
                    max(heated[i] for ids in thermal.die_blocks[die].values() for i in ids)
                    for die in range(thermal.ndie)
                ])
                np.testing.assert_allclose(thermal.die_max(), expected_die_max, rtol=0, atol=1e-12)
                self.assertIsNone(thermal.energy_residual_j)
            finally:
                thermal.close()


if __name__ == "__main__":
    unittest.main()
