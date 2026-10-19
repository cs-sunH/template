"""Small regression for the retained action-energy adapter."""
import json
from pathlib import Path
import unittest
from wafercosim.action_energy import EnergyModel

ROOT = Path(__file__).resolve().parents[1]

class EnergyCoreTests(unittest.TestCase):
    def test_fallback_coefficients_are_units_and_destinations(self):
        config = json.loads((ROOT / "configs/fixtures/tiny-analytical.json").read_text())
        model = EnergyModel(config)
        self.assertAlmostEqual(model.energy("pe", 10, 300), 2e-12)
        self.assertAlmostEqual(model.energy("vec", 10, 300), 4e-11)
        self.assertEqual(model.thermal_destination("hbm"), "hbm")
        self.assertEqual(model.thermal_destination("dram"), "external")
        self.assertGreater(model.background("compute_leak", 1, 350), 0.0)

if __name__ == "__main__":
    unittest.main()
