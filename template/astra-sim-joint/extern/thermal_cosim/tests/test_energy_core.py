"""Small regression for the retained action-energy adapter."""
import json
from pathlib import Path
import unittest
from wafercosim.action_energy import EnergyCalibrationError, EnergyModel
from wafercosim.paths import config_path

ROOT = Path(__file__).resolve().parents[1]

class EnergyCoreTests(unittest.TestCase):
    @staticmethod
    def _table():
        return json.loads(config_path("energy", "research-v1.json").read_text())

    @staticmethod
    def _config(*, strict=True):
        return {
            "calibration": {
                "energy_table": "configs/energy/research-v1.json",
                "strict": strict,
                "accept_assumed_energy": True,
                "memory_organization_sensitivity_accept": {"hbm": True, "dram": True},
            },
            "thermal": {"initial_k": 300.0},
        }

    @classmethod
    def _hardware_facts(cls):
        # Synthetic matching test fixture, never a hardware export for a host.
        domain = cls._table()["hardware_domain"]
        sram = cls._table()["background"]["sram_leak"]
        noc = domain["noc"]
        topology = noc["supported_topologies"][0]
        return {
            "sram_capacity_bytes_per_die": int(sram["macro_capacity_bytes"] * sram["macro_count_per_die"]),
            "pe_ops_s_per_die": domain["compute"]["pe_ops_s_per_die"],
            "vec_ops_s_per_die": domain["compute"]["vec_ops_s_per_die"],
            "hbm_capacity_bytes_per_die": domain["hbm"]["configured_capacity_bytes_per_die"],
            "dram_capacity_bytes": domain["dram"]["configured_capacity_bytes"],
            "compute": {
                "weight_bytes": domain["compute"]["weight_bytes"],
                "activation_bytes": domain["compute"]["activation_bytes"],
            },
            "noc": {
                "nx": topology["nx"],
                "ny": topology["ny"],
                "flit_width_bits": noc["flit_width_bits"],
                "num_vcs": noc["num_vcs"],
                "packet_size_flits": noc["packet_size_flits"],
                "router": noc["router"],
                "routing": noc["routing"],
            },
            "hbm": {"protocol": domain["hbm"]["protocol"], "config": domain["hbm"]["config"]},
            "dram": {"protocol": domain["dram"]["protocol"], "config": domain["dram"]["config"]},
            "sram": {
                "macro_capacity_bytes": sram["macro_capacity_bytes"],
                "macro_count_per_die": sram["macro_count_per_die"],
                "word_width_bits": sram["source"]["word_bits"],
                "access_bytes": sram["source"]["access_bytes"],
            },
        }

    def test_fallback_coefficients_are_units_and_destinations(self):
        config = json.loads(config_path("fixtures", "tiny-analytical.json").read_text())
        model = EnergyModel(config)
        self.assertAlmostEqual(model.energy("pe", 10, 300), 2e-12)
        self.assertAlmostEqual(model.energy("vec", 10, 300), 4e-11)
        self.assertEqual(model.thermal_destination("hbm"), "hbm")
        self.assertEqual(model.thermal_destination("dram"), "external")
        self.assertGreater(model.background("compute_leak", 1, 350), 0.0)

    def test_table_mode_requires_explicit_host_facts_even_when_nonstrict(self):
        config = self._config(strict=False)
        with self.assertRaisesRegex(EnergyCalibrationError, "hardware_facts is required"):
            EnergyModel(config)

    def test_legacy_hardware_and_service_table_inputs_are_rejected(self):
        for section, key in (("top", "hardware"), ("calibration", "hardware"), ("calibration", "service_table")):
            config = self._config()
            target = config if section == "top" else config[section]
            target[key] = {}
            with self.subTest(section=section, key=key):
                with self.assertRaisesRegex(EnergyCalibrationError, "legacy"):
                    EnergyModel(config, hardware_facts=self._hardware_facts())
        for key in ("model", "workload", "simulation", "compute", "mapping", "performance"):
            config = self._config()
            config[key] = {}
            with self.subTest(key=key):
                with self.assertRaisesRegex(EnergyCalibrationError, "legacy"):
                    EnergyModel(config, hardware_facts=self._hardware_facts())

    def test_host_capacity_and_precision_mismatches_fail_strict_domain(self):
        config = self._config()
        facts = self._hardware_facts()
        facts["hbm_capacity_bytes_per_die"] += 1
        with self.assertRaisesRegex(EnergyCalibrationError, "hbm_capacity_bytes_per_die"):
            EnergyModel(config, hardware_facts=facts)

        facts = self._hardware_facts()
        facts["compute"]["weight_bytes"] *= 2
        with self.assertRaisesRegex(EnergyCalibrationError, "compute.weight_bytes"):
            EnergyModel(config, hardware_facts=facts)

    def test_noc_topology_comes_from_host_facts_not_thermal_geometry(self):
        config = self._config()
        config["geometry"] = {"die_columns": 8, "die_rows": 6}
        facts = self._hardware_facts()
        facts["noc"]["nx"], facts["noc"]["ny"] = 4, 2

        model = EnergyModel(config, hardware_facts=facts)

        self.assertEqual(model.metadata()["selected_noc_topology"], "4x2")

    def test_sram_leakage_uses_consistent_host_macro_population(self):
        config = self._config()
        facts = self._hardware_facts()
        row = self._table()["background"]["sram_leak"]
        reference_count = row["macro_count_per_die"]
        actual_count = int(reference_count) + 17
        macro_bytes = row["macro_capacity_bytes"]
        facts["sram"]["macro_count_per_die"] = actual_count
        facts["sram_capacity_bytes_per_die"] = macro_bytes * actual_count

        model = EnergyModel(config, hardware_facts=facts)
        actual_watts = model.background("sram_leak", 1, 300)
        expected_watts = model._interpolate_rows(
            row["watts_per_macro_by_temperature_k"], 300, "watts_per_macro"
        ) * actual_count

        self.assertEqual(actual_watts, expected_watts)
        self.assertNotEqual(actual_watts, expected_watts / actual_count * reference_count)
        summary = model.metadata()["hardware_fact_summary"]
        self.assertEqual(summary["sram_actual_macro_count_per_die"], actual_count)
        self.assertEqual(summary["sram_reference_macro_count_per_die"], reference_count)
        self.assertEqual(
            model.metadata()["hardware_facts_sha256"],
            EnergyModel(config, hardware_facts=facts).metadata()["hardware_facts_sha256"],
        )

    def test_sram_capacity_count_and_macro_organization_are_fail_closed(self):
        config = self._config(strict=False)
        facts = self._hardware_facts()
        facts["sram"]["macro_count_per_die"] += 1
        with self.assertRaisesRegex(EnergyCalibrationError, "must equal"):
            EnergyModel(config, hardware_facts=facts)

        facts = self._hardware_facts()
        facts["sram"]["word_width_bits"] *= 2
        with self.assertRaisesRegex(EnergyCalibrationError, "CACTI organization"):
            EnergyModel(config, hardware_facts=facts)

if __name__ == "__main__":
    unittest.main()
