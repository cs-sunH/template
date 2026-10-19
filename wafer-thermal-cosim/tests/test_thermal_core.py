"""Core low-order thermal contract tests kept in the trimmed repository."""
import json
import unittest
from pathlib import Path
import numpy as np
from wafercosim.thermal import LiquidRC

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/thermal/3dice_8die_reference.json"


def config():
    return json.loads(CONFIG.read_text())


class LiquidRCTests(unittest.TestCase):
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
