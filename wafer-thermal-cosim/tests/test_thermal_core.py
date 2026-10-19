"""Core low-order thermal contract tests kept in the trimmed repository."""
import json
import unittest
from pathlib import Path
import numpy as np
from wafercosim.backends import create_thermal
from wafercosim.thermal import LiquidRC

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/thermal/3dice_8die_reference.json"


def config():
    return json.loads(CONFIG.read_text())


class LiquidRCTests(unittest.TestCase):
    def test_factory_constructs_rc_without_execution_configuration(self):
        thermal = create_thermal(config(), ROOT)
        self.assertIsInstance(thermal, LiquidRC)
        self.assertEqual(thermal.ndie, 8)
        self.assertEqual(thermal.nsolid, 48)

    def test_host_supplied_block_footprints_are_preserved(self):
        settings = config()
        settings["geometry"]["hbm"] = {
            "x_m": 0.001, "y_m": 0.019, "width_m": 0.015, "height_m": 0.005,
        }
        settings["geometry"]["router"] = {
            "x_m": 0.018, "y_m": 0.002, "width_m": 0.002, "height_m": 0.015,
        }
        thermal = LiquidRC(settings)
        for kind in ("hbm", "router"):
            rectangle = settings["geometry"][kind]
            for die in (0, thermal.ndie - 1):
                block = thermal.blocks[thermal.die_blocks[die][kind][0]]
                self.assertAlmostEqual(block.x,
                    (die % thermal.nx) * settings["geometry"]["module_width_m"] + rectangle["x_m"])
                self.assertAlmostEqual(block.y,
                    (die // thermal.nx) * settings["geometry"]["module_height_m"] + rectangle["y_m"])
                self.assertEqual((block.width, block.height), (rectangle["width_m"], rectangle["height_m"]))

    def test_legacy_execution_inputs_are_rejected(self):
        for name in ("hardware", "model", "workload", "simulation", "compute"):
            with self.subTest(name=name):
                settings = config()
                settings[name] = {}
                with self.assertRaisesRegex(ValueError, "execution inputs belong to the joint host"):
                    LiquidRC(settings)
        settings = config()
        settings["thermal"]["throttle_ratio"] = 0.6
        with self.assertRaisesRegex(ValueError, "unused throttle settings"):
            LiquidRC(settings)
        for name in ("mapping", "performance"):
            with self.subTest(name=name):
                settings = config()
                settings[name] = {}
                with self.assertRaisesRegex(ValueError, "unsupported thermal config fields"):
                    LiquidRC(settings)

    def test_invalid_geometry_is_rejected_before_constructing_network(self):
        cases = (
            ("die_columns", True), ("die_rows", 0),
            ("compute_groups_per_side", 1.5), ("compute_side_m", float("nan")),
            ("compute_side_m", 1.0),
            ("hbm", {"x_m": 0, "y_m": 0, "width_m": 0, "height_m": 0.01}),
            ("router", {"x_m": 1.0, "y_m": 0, "width_m": 0.001, "height_m": 0.01}),
        )
        for name, value in cases:
            with self.subTest(name=name):
                settings = config()
                settings["geometry"][name] = value
                with self.assertRaises(ValueError):
                    LiquidRC(settings)

    def test_zero_power_equilibrium(self):
        thermal = LiquidRC(config())
        thermal.step(np.zeros(thermal.nsolid), 0.01)
        np.testing.assert_allclose(thermal.T, 300.0, atol=1e-8)

    def test_energy_conservation_and_spatial_coupling(self):
        thermal = LiquidRC(config())
        power = np.zeros(thermal.nsolid)
        power[thermal.die_blocks[0]["compute"][0]] = 200.0
        for _ in range(100):
            thermal.step(power, 0.01)
        self.assertLess(abs(thermal.energy_residual_j), 1e-6)
        self.assertGreater(thermal.die_max()[0], 300.0)
        self.assertGreater(thermal.T[thermal.nsolid], 300.0)
        flow_test = LiquidRC(config())
        uniform = np.full(flow_test.nsolid, 20.0)
        for _ in range(100):
            flow_test.step(uniform, 0.01)
        self.assertGreater(flow_test.T[flow_test.nsolid + 1], flow_test.T[flow_test.nsolid])

    def test_step_refinement_is_bounded(self):
        coarse = LiquidRC(config())
        fine = LiquidRC(config())
        power = np.full(coarse.nsolid, 20.0)
        for _ in range(100):
            coarse.step(power, 0.01)
        for _ in range(200):
            fine.step(power, 0.005)
        self.assertLess(float(np.max(np.abs(coarse.T - fine.T))), 0.4)


if __name__ == "__main__":
    unittest.main()
