"""Independent small allocation expectations and archived policy provenance."""
from fractions import Fraction
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE / "validation"))
import compiled_gemm_oracle as legacy
import grouped_gemm_oracle as grouped


def activity(owner, *, gate=0, hbm0=0, hbm1=0, local=0, private=0):
    work = [0.0] * len(legacy.RESOURCE_NAMES)
    for name, value in (("COMPUTE_GATE", gate), ("HBM0", hbm0),
                        ("HBM1", hbm1), ("LOCAL", local)):
        work[legacy.RESOURCE_IDS[name]] = value
    return SimpleNamespace(request=owner, work=work, private_time_s=private)


def capacities(hbm0=1, hbm1=1):
    result = [1e30] * len(legacy.RESOURCE_NAMES)
    result[legacy.RESOURCE_IDS["COMPUTE_GATE"]] = 1
    result[legacy.RESOURCE_IDS["HBM0"]] = hbm0
    result[legacy.RESOURCE_IDS["HBM1"]] = hbm1
    return result


class GroupedOracleTest(unittest.TestCase):
    def assert_rates(self, actual, expected):
        self.assertEqual(len(actual), len(expected))
        for got, wanted in zip(actual, expected):
            self.assertAlmostEqual(got, float(wanted), delta=1e-12)

    def test_owner_stage_membership_changes_compute_shares(self):
        active = [activity(0, gate=10), activity(0, hbm0=2),
                  activity(1, gate=10)]
        rates = grouped.grouped_rates(active, capacities(), [1, 2, 1])
        self.assert_rates(rates, [Fraction(1, 40), Fraction(1, 2), Fraction(3, 40)])
        self.assert_rates(legacy.fill_rates(active, capacities()),
                          [Fraction(1, 20), Fraction(1, 2), Fraction(1, 20)])

    def test_owner_sums_stay_fixed_across_lp_tiers(self):
        active = [activity(0, gate=10), activity(0, hbm0=2),
                  activity(0, hbm0=1), activity(1, gate=10),
                  activity(1, hbm0=5)]
        self.assert_rates(grouped.grouped_rates(active, capacities(), [1, 2, 1, 1, 5]),
                          [Fraction(3, 50), Fraction(3, 19), Fraction(3, 19),
                           Fraction(1, 25), Fraction(2, 19)])

    def test_equivalent_same_owner_fragment_split_keeps_service(self):
        merged = [activity(0, hbm0=4), activity(0, gate=10),
                  activity(1, hbm0=2), activity(1, gate=10)]
        split = [activity(0, hbm0=1), activity(0, hbm0=3), activity(0, gate=10),
                 activity(1, hbm0=2), activity(1, gate=10)]
        m = grouped.grouped_rates(merged, capacities(), [4, 1, 2, 1])
        s = grouped.grouped_rates(split, capacities(), [1, 3, 1, 2, 1])
        self.assertAlmostEqual(4 * m[0], s[0] + 3 * s[1], delta=1e-12)
        self.assert_rates([m[1], m[2], m[3]], [s[2], s[3], s[4]])

    def test_disabled_resource_activity_does_not_keep_owner_share(self):
        active = [activity(0, gate=10), activity(0, hbm0=2),
                  activity(1, gate=10)]
        self.assert_rates(grouped.grouped_rates(active, capacities(hbm0=0), [1, 2, 1]),
                          [Fraction(1, 20), 0, Fraction(1, 20)])

    def test_private_bound_freezes_one_compute(self):
        active = [activity(0, gate=10, private=100), activity(0, hbm0=2),
                  activity(1, gate=10, private=10)]
        self.assert_rates(grouped.grouped_rates(active, capacities(), [1, 2, 1]),
                          [Fraction(1, 100), Fraction(1, 2), Fraction(9, 100)])

    def test_nonbinding_local_work_still_defines_canonical_quantity(self):
        active = [activity(0, gate=10, local=100), activity(0, hbm0=2),
                  activity(1, gate=10)]
        self.assert_rates(grouped.grouped_rates(active, capacities(), [100, 2, 1]),
                          [Fraction(1, 1030), Fraction(1, 2), Fraction(51, 515)])

    def test_archived_compiled_fixture_preserves_both_policies_and_limits(self):
        raw = json.loads((SOURCE / "docs/analysis/runtime-waves-20261013/"
                          "bounded-lifecycle-v1/raw-fixture.json").read_text())
        waves, expected = legacy.make_waves(raw)
        self.assertEqual(sum(w.compute_local_bytes for w in waves), 9636640)
        self.assertEqual(expected["local_bus_bytes"], 9636640)
        old = legacy.build_report(raw)
        new = grouped.build_report(raw)
        self.assertEqual(new["legacy_equal_progress_comparisons"], old["comparisons"])
        self.assertEqual(new["acceptance_limits"], old["acceptance_limits"])
        self.assertEqual(new["status"], "failed")
        self.assertEqual(new["available_granularity_levels"]["selected_minimum_E"], None)
        offset = new["comparisons"]["two_comp"][1]["independent_discrete_oracle"]
        self.assertAlmostEqual(offset["finish_s_by_request"]["comp0"],
                               0.4261738759606149, delta=1e-10)

    def test_invalid_quantities_are_rejected(self):
        active = [activity(0, gate=1)]
        for q in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                grouped.grouped_rates(active, capacities(), [q])


if __name__ == "__main__":
    unittest.main()
