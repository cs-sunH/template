from types import SimpleNamespace
import unittest

from online.graph_batch_builder import CreditGateRange, CreditGateSet
from online.sh30_online_scheduler import Sh30OnlineScheduler


class CompactCreditGateRangeTests(unittest.TestCase):
    def test_large_decode_gate_extent_stays_one_run(self):
        scheduler = Sh30OnlineScheduler.__new__(Sh30OnlineScheduler)
        runtime = SimpleNamespace(
            request_id="req",
            remote_read_credit_plan={"tiles_per_pass": 100_000})
        blocks = scheduler._remote_credit_body_blocks(
            {"pass_spans": [object()]}, [(1, 0)], [(runtime, 1)],
            start_iter=1, end_iter=1)

        gates = blocks[0]["gates"]
        self.assertIsInstance(gates, CreditGateSet)
        self.assertEqual(len(gates), 100_000)
        self.assertEqual(len(gates.ranges), 1)
        self.assertEqual(gates[0], ("req", 1))
        self.assertEqual(gates[-1], ("req", 100_000))
        self.assertEqual(
            gates,
            [("req", tile) for tile in range(1, 100_001)])

    def test_gate_set_concatenates_distinct_requests_without_expansion(self):
        gates = CreditGateSet((
            CreditGateRange("a", 1, 3),
            CreditGateRange("b", 7, 2),
        ))
        self.assertEqual(len(gates.ranges), 2)
        self.assertEqual(list(gates), [
            ("a", 1), ("a", 2), ("a", 3), ("b", 7), ("b", 8)])
        self.assertEqual(gates.request_ids(), frozenset(("a", "b")))
        self.assertTrue(gates.contains("a", 2))
        self.assertFalse(gates.contains("a", 7))


if __name__ == "__main__":
    unittest.main()
