"""Contract, power-projection, determinism and state-isolation tests (plan V03-V06)."""
import unittest

from fixtures import reference_session, reference_thermal_config, synthetic_hardware
from wafercosim.contracts import (
    ActivityWindow,
    ContractError,
    MappingSnapshot,
)
from wafercosim.thermal import LiquidRC

NS_PER_S = 1_000_000_000


def _window(session, seq, start_ns, end_ns, actions=(), dwell=(), **kw):
    return ActivityWindow(
        run_id=session.run_id,
        activity_seq=seq,
        start_ns=start_ns,
        end_ns=end_ns,
        hardware_hash=session.hardware.content_hash,
        mapping_epoch=session.mapping.mapping_epoch,
        resource_epoch=kw.get("resource_epoch", 1),
        actions=actions,
        dwell=dwell,
        coverage=kw.get("coverage", "measured"),
        provenance=kw.get("provenance", "test"),
    )


class ContractTests(unittest.TestCase):
    def test_mapping_weights_must_sum_to_one(self):
        thermal = LiquidRC(reference_thermal_config())
        hardware = synthetic_hardware()
        compute0 = [b.name for b in thermal.blocks if b.die == 0 and b.kind == "compute"]
        with self.assertRaisesRegex(ContractError, "weights sum to"):
            MappingSnapshot.build(
                mapping_epoch="e",
                hardware_hash=hardware.content_hash,
                source_block_weights={"die0.compute": {name: 0.5 for name in compute0}},
                rank_to_die={0: 0},
                block_ids=[b.name for b in thermal.blocks],
            )

    def test_mapping_rejects_unknown_thermal_block(self):
        with self.assertRaisesRegex(ContractError, "unknown thermal block"):
            MappingSnapshot.build(
                mapping_epoch="e",
                hardware_hash="h",
                source_block_weights={"die0.compute": {"nope": 1.0}},
                rank_to_die={0: 0},
                block_ids=["d0_c0"],
            )

    def test_activity_window_requires_forward_interval(self):
        with self.assertRaisesRegex(ContractError, "end_ns > start_ns"):
            ActivityWindow(
                run_id="r", activity_seq=0, start_ns=10, end_ns=10,
                hardware_hash="h", mapping_epoch="m", resource_epoch=1, actions=(),
            )

    def test_unregistered_action_is_reported_not_zeroed(self):
        with self.assertRaisesRegex(ContractError, "not a registered action"):
            ActivityWindow(
                run_id="r", activity_seq=0, start_ns=0, end_ns=10,
                hardware_hash="h", mapping_epoch="m", resource_epoch=1,
                actions=(("die0.compute", "made_up_op", 1.0),),
            )


class PowerLedgerTests(unittest.TestCase):
    def test_energy_ledger_conserves_and_splits_external(self):
        session = reference_session()
        window = _window(
            session, 0, 0, 10_000_000,  # 10 ms
            actions=(
                ("die0.compute", "pe", 1.0e9),
                ("die0.compute", "sram_read", 1.0e9),
                ("die0.hbm", "hbm_read", 1.0e8),
                ("die0.router", "noc", 1.0e7),
                ("external.dram", "dram_read", 1.0e7),
            ),
            dwell=(("die0.compute", 9_000_000, 1_000_000),),
        )
        power = session.estimate_power(window)
        dt_s = window.dt_s
        rebuilt = sum(w * dt_s for w in power.power_w) + power.external_j
        self.assertAlmostEqual(rebuilt, power.total_energy_j, places=6, msg="sum(P)*dt + external != total")
        dynamic = sum(power.dynamic_j.values())
        background = sum(power.background_j.values())
        self.assertAlmostEqual(dynamic + background + power.external_j, power.total_energy_j, places=6)
        self.assertGreater(power.external_j, 0.0)
        self.assertGreater(max(power.power_w), 0.0)

    def test_idle_window_advances_leakage_without_dynamic_energy(self):
        session = reference_session()
        window = _window(session, 0, 0, 10_000_000)
        power = session.estimate_power(window)
        self.assertAlmostEqual(sum(power.dynamic_j.values()), 0.0, places=12)
        self.assertGreater(sum(power.background_j.values()), 0.0)

    def test_source_without_mapping_is_not_averaged_across_dies(self):
        session = reference_session()
        window = _window(session, 0, 0, 1_000_000,
                         actions=(("die0.gpu", "pe", 1.0),))
        with self.assertRaisesRegex(ContractError, "no explicit"):
            session.estimate_power(window)


class AdvanceAndIsolationTests(unittest.TestCase):
    def test_advance_requires_contiguous_windows(self):
        session = reference_session()
        power = session.estimate_power(_window(session, 0, 0, 1_000_000))
        feedback = session.advance_committed(power)
        self.assertEqual(session.committed_time_ns, 1_000_000)
        self.assertEqual(feedback.effective_ns, 1_000_000)
        self.assertEqual(feedback.thermal_epoch, 1)

    def test_gap_and_overlap_rejected(self):
        session = reference_session()
        session.advance_committed(session.estimate_power(_window(session, 0, 0, 1_000_000)))
        gap = session.estimate_power(_window(session, 1, 2_000_000, 3_000_000))
        with self.assertRaisesRegex(ContractError, "unexplained gap"):
            session.advance_committed(gap)
        # an explicit idle window fills the gap
        idle = session.estimate_power(session.idle_window(dt_ns=1_000_000, activity_seq=1))
        session.advance_committed(idle)
        self.assertEqual(session.committed_time_ns, 2_000_000)

    def test_duplicate_commit_is_idempotent(self):
        session = reference_session()
        window = _window(session, 0, 0, 1_000_000,
                         actions=(("die0.compute", "pe", 1.0e9),))
        power = session.estimate_power(window)
        first = session.advance_committed(power)
        temperature_after = session.block_temperature_k()
        second = session.advance_committed(power)
        self.assertEqual(first.effective_ns, second.effective_ns)
        self.assertEqual(first.thermal_epoch, second.thermal_epoch)
        self.assertEqual(temperature_after, session.block_temperature_k())

    def test_fork_isolated_from_committed_state(self):
        session = reference_session()
        session.advance_committed(session.estimate_power(
            _window(session, 0, 0, 1_000_000,
                    actions=(("die0.compute", "pe", 5.0e9),))))
        committed_temperature = session.block_temperature_k()
        committed_time = session.committed_time_ns
        committed_energy = sum(session.thermal.feedback()["block_k"].values())

        branch = session.fork()
        branch.advance_committed(branch.estimate_power(
            _window(branch, 0, 1_000_000, 2_000_000,
                    actions=(("die0.compute", "pe", 1.0e12),))))
        self.assertNotEqual(branch.block_temperature_k(), committed_temperature)
        self.assertEqual(branch.committed_time_ns, 2_000_000)

        # the committed session is untouched
        self.assertEqual(session.block_temperature_k(), committed_temperature)
        self.assertEqual(session.committed_time_ns, committed_time)
        self.assertEqual(sum(session.thermal.feedback()["block_k"].values()), committed_energy)

    def test_reset_restores_initial_state(self):
        session = reference_session()
        session.advance_committed(session.estimate_power(
            _window(session, 0, 0, 1_000_000,
                    actions=(("die0.compute", "pe", 5.0e9),))))
        session.reset()
        self.assertEqual(session.committed_time_ns, 0)
        self.assertEqual(session.thermal_epoch, 0)
        self.assertEqual(session.state_version, 0)

    def test_replay_is_deterministic(self):
        def run():
            session = reference_session()
            for seq, (start, end) in enumerate([(0, 1_000_000), (1_000_000, 2_000_000)]):
                session.advance_committed(session.estimate_power(_window(
                    session, seq, start, end,
                    actions=(("die0.compute", "pe", 2.0e9),))))
            return session.block_temperature_k()

        self.assertEqual(run(), run())


class ThermalInputValidationTests(unittest.TestCase):
    def test_step_rejects_non_finite_and_negative(self):
        thermal = LiquidRC(reference_thermal_config())
        with self.assertRaises(ValueError):
            thermal.step([1.0] * thermal.nsolid, 0.0)
        with self.assertRaises(ValueError):
            thermal.step([1.0] * thermal.nsolid, float("nan"))
        with self.assertRaises(ValueError):
            thermal.step([-1.0] * thermal.nsolid, 0.01)
        with self.assertRaises(ValueError):
            thermal.step([float("inf")] * thermal.nsolid, 0.01)

    def test_snapshot_carries_coolant_and_ledger(self):
        thermal = LiquidRC(reference_thermal_config())
        thermal.step([10.0] * thermal.nsolid, 0.01)
        snap = thermal.snapshot()
        self.assertEqual(snap["temperature_k"].shape[0], thermal.nsolid + thermal.ndie)
        self.assertGreater(snap["input_j"], 0.0)
        self.assertEqual(snap["state_version"], 1)


if __name__ == "__main__":
    unittest.main()
