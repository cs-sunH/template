"""Joint-side thermal bridge / policy tests (plan V12, V13 and closed-loop plumbing).

These exercise the *plumbing* between joint facts and the embedded thermal core
with the packaged synthetic fixture.  They are engineering checks and say
nothing about target-device power/thermal accuracy.
"""
import json
import sys
import unittest
from pathlib import Path

_JOINT_DIR = Path(__file__).resolve()
_REPO_ROOT = _JOINT_DIR.parents[4]
_THERMAL_DIR = _REPO_ROOT / "extern" / "thermal_cosim"
if str(_THERMAL_DIR) not in sys.path:
    sys.path.insert(0, str(_THERMAL_DIR))

from wafercosim.contracts import ActivityWindow, ContractError  # noqa: E402
from wafercosim.fixtures import reference_thermal_config, synthetic_thermal_facts_export  # noqa: E402

from joint.thermal_bridge import (  # noqa: E402
    CoarseActivitySampler,
    CoarseCapacityInstaller,
    SamplerState,
    ThermalBridge,
    ThermalRunConfig,
    build_fixture_mapping,
    hardware_snapshot_from_facts,
    synthetic_rank_to_die,
)
from joint.thermal_policy import ThermalPolicy, ThermalPolicyConfig  # noqa: E402

LLAMA_DIR = _REPO_ROOT / "sh_test_mesh" / "workload" / "llama2_7b_inference"
POLICY_PATH = LLAMA_DIR / "thermal" / "synthetic_policy.json"
THERMAL_DIE_COUNT = 8


class _FakeRuntime:
    """Records the capacity updates a closed-loop install makes (no C++ needed)."""

    def __init__(self, rank: int = 0):
        self.rank = rank
        self.calls: list[tuple[int, float, int]] = []

    def update_capacity(self, resource: int, rate: float, now_ns: int) -> None:
        self.calls.append((int(resource), float(rate), int(now_ns)))


def _installer() -> tuple["CoarseCapacityInstaller", _FakeRuntime]:
    runtime = _FakeRuntime()
    installer = CoarseCapacityInstaller(
        runtime, domain_to_rank={"die0": 0, "die1": 1, "die2": 2, "die3": 3})
    return installer, runtime


def _hardware_mapping_policy():
    hardware = hardware_snapshot_from_facts(synthetic_thermal_facts_export())
    thermal_config = reference_thermal_config()
    mapping = build_fixture_mapping(
        hardware, thermal_config,
        mapping_epoch="thermal-fixture-v1",
        rank_to_die=synthetic_rank_to_die(54, THERMAL_DIE_COUNT),
    )
    policy = ThermalPolicy(ThermalPolicyConfig.from_json(
        json.loads(POLICY_PATH.read_text())))
    return hardware, mapping, policy


def _make_bridge(mode: str, run_id: str = "r"):
    """Build a bridge; closed_loop gets a recording runtime installer."""
    hardware, mapping, policy = _hardware_mapping_policy()
    installer, runtime = _installer() if mode == "closed_loop" else (None, None)
    bridge = ThermalBridge(
        run_id=run_id, config=_run_config(mode), hardware=hardware,
        mapping=mapping, policy=policy, installer=installer)
    return bridge, runtime


def _run_config(mode, *, policy=POLICY_PATH):
    return ThermalRunConfig(
        mode=mode,
        thermal_config_path="package:thermal/3dice_8die_reference.json",
        energy_table_path="configs/energy/research-v1.json",
        energy_strict=True,
        policy_path=str(policy) if policy else None,
        mapping_epoch="thermal-fixture-v1",
        thermal_step_ns=10_000_000,
        base_frequency_hz=1.0e9,
        accept_assumed_energy=True,
    )


def _hot_window(bridge, seq, start_ns, end_ns, *, pe_ops):
    return ActivityWindow(
        run_id=bridge.run_id,
        activity_seq=seq,
        start_ns=start_ns,
        end_ns=end_ns,
        hardware_hash=bridge.hardware.content_hash,
        mapping_epoch=bridge.mapping.mapping_epoch,
        resource_epoch=bridge.resource_epoch,
        actions=(
            ("die0.compute", "pe", pe_ops),
            ("die0.compute", "sram_read", pe_ops),
        ),
        dwell=(("die0.compute", end_ns - start_ns, 0),),
        coverage="measured",
        provenance="test-hot-window",
    )


class BridgeContractTests(unittest.TestCase):
    def test_off_mode_has_no_session(self):
        bridge, _ = _make_bridge("off")
        self.assertFalse(bridge.enabled)
        self.assertIsNone(bridge.ingest(_hot_window(bridge, 0, 0, 10_000_000, pe_ops=1)))

    def test_closed_loop_requires_explicit_mapping(self):
        hardware = hardware_snapshot_from_facts(synthetic_thermal_facts_export())
        from wafercosim.contracts import MappingSnapshot

        empty = MappingSnapshot.build(
            mapping_epoch="m", hardware_hash=hardware.content_hash,
            source_block_weights={"die0.compute": {"d0_c0": 1.0}},
            rank_to_die={}, block_ids=["d0_c0"], external_sinks=[])
        with self.assertRaisesRegex(ContractError, "explicit rank"):
            ThermalBridge(run_id="r", config=_run_config("closed_loop"),
                          hardware=hardware, mapping=empty,
                          policy=ThermalPolicy(ThermalPolicyConfig.from_json(
                              json.loads(POLICY_PATH.read_text()))))

    def test_closed_loop_requires_an_installer(self):
        hardware, mapping, policy = _hardware_mapping_policy()
        bridge = ThermalBridge(run_id="r", config=_run_config("closed_loop"),
                               hardware=hardware, mapping=mapping, policy=policy)
        with self.assertRaisesRegex(ContractError, "capacity installer"):
            bridge.ingest(_hot_window(bridge, 0, 0, 10_000_000, pe_ops=4.0e10))

    def test_policy_requires_declared_reference(self):
        from joint.thermal_bridge import load_policy

        with self.assertRaisesRegex(ContractError, "policy reference is required"):
            load_policy(_run_config("shadow", policy=None))


class ShadowClosedLoopTests(unittest.TestCase):
    def _drive(self, mode):
        bridge, runtime = _make_bridge(mode)
        results = []
        for seq in range(6):
            result = bridge.ingest(_hot_window(
                bridge, seq, seq * 10_000_000, (seq + 1) * 10_000_000,
                pe_ops=4.0e10))
            results.append(result)
        return bridge, runtime, results

    def test_shadow_produces_cap_without_installing(self):
        bridge, runtime, results = self._drive("shadow")
        self.assertEqual(bridge.resource_epoch, 1)
        envelopes = [r["envelope"] for r in results]
        self.assertTrue(any(e.domain_fraction("die0") == 0.5 for e in envelopes),
                        "hot fixture did not drive die0 to the de-rated step")
        for envelope in envelopes:
            for row in envelope.domains:
                self.assertFalse(row["applied"], "shadow must not install caps")
        # no runtime handle was ever touched in shadow mode
        self.assertIsNone(runtime)

    def test_closed_loop_installs_cap_into_the_runtime(self):
        bridge, runtime, results = self._drive("closed_loop")
        de_rated = [r for r in results if r["envelope"].domain_fraction("die0") == 0.5]
        self.assertTrue(de_rated)
        for result in de_rated:
            apex = next(row for row in result["envelope"].domains
                        if row["domain"] == "die0")
            self.assertTrue(apex["applied"], "closed_loop must mark the cap applied")
        self.assertGreater(bridge.resource_epoch, 1)
        # the actual capacity update reached the runtime ABI (COMPUTE_GATE, 0.5)
        gate_updates = [c for c in runtime.calls if c[0] == 0]
        self.assertTrue(any(abs(rate - 0.5) < 1e-12 for _, rate, _ in gate_updates),
                        f"no 0.5 COMPUTE_GATE install recorded: {runtime.calls}")

    def test_audit_rows_are_written(self):
        bridge, _ = _make_bridge("shadow")
        bridge.ingest(_hot_window(bridge, 0, 0, 10_000_000, pe_ops=1.0e10))
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            written = bridge.write_audit(Path(directory))
            names = {path.name for path in written}
        self.assertIn("resolved_hardware.json", names)
        self.assertIn("physical_mapping.json", names)
        self.assertIn("thermal_manifest.json", names)
        self.assertIn("power_windows.jsonl", names)
        self.assertIn("thermal_feedback.jsonl", names)
        self.assertIn("resource_envelopes.jsonl", names)


class ActivitySamplerTests(unittest.TestCase):
    def _counters(self, now_ns, int4, vec, hbm_read, hbm_write, busy_ns):
        return {
            "now_ns": now_ns,
            "accounting_history_complete": True,
            "compute_busy_ns": busy_ns,
            "served_by_category": {"comp": {"int4_ops": int4, "vec_eqops": vec}},
            "hbm_served_bytes_by_row_direction": [[hbm_read, hbm_write]],
        }

    def test_delta_yields_measured_actions(self):
        sampler = CoarseActivitySampler(
            run_id="r", hardware_hash="h", mapping_epoch="m",
            rank_to_source={0: "die0"})
        first = SamplerState.from_counters(
            self._counters(0, 100.0, 10.0, 1000.0, 0.0, 5_000_000))
        second = SamplerState.from_counters(
            self._counters(10_000_000, 500.0, 40.0, 3000.0, 500.0, 8_000_000))
        window = sampler.window(0, first, second, resource_epoch=1)
        actions = {(source, kind): count for source, kind, count in window.actions}
        self.assertEqual(actions[("die0.compute", "pe")], 400.0)
        self.assertEqual(actions[("die0.compute", "vec")], 30.0)
        self.assertEqual(actions[("die0.hbm", "hbm_read")], 2000.0)
        self.assertEqual(actions[("die0.hbm", "hbm_write")], 500.0)
        self.assertEqual(window.dwell[0][1], 3_000_000)  # active ns
        self.assertEqual(window.dwell[0][2], 7_000_000)  # idle ns
        self.assertEqual(window.coverage, "measured")

    def test_incomplete_accounting_is_refused(self):
        counters = self._counters(0, 1.0, 0.0, 0.0, 0.0, 0.0)
        counters["accounting_history_complete"] = False
        with self.assertRaisesRegex(ContractError, "incomplete"):
            SamplerState.from_counters(counters)

    def test_counters_must_be_monotonic(self):
        sampler = CoarseActivitySampler(
            run_id="r", hardware_hash="h", mapping_epoch="m",
            rank_to_source={0: "die0"})
        high = SamplerState.from_counters(
            self._counters(10_000_000, 500.0, 0.0, 0.0, 0.0, 0.0))
        low = SamplerState.from_counters(
            self._counters(20_000_000, 100.0, 0.0, 0.0, 0.0, 0.0))
        with self.assertRaisesRegex(ContractError, "backwards"):
            sampler.window(0, high, low, resource_epoch=1)

    def test_sampled_window_drives_the_bridge_end_to_end(self):
        """Real counters -> ActivityWindow -> power -> temperature -> cap."""
        bridge, _ = _make_bridge("shadow")
        sampler = CoarseActivitySampler(
            run_id=bridge.run_id, hardware_hash=bridge.hardware.content_hash,
            mapping_epoch=bridge.mapping.mapping_epoch,
            rank_to_source={0: "die0"})

        def counters(now_ns, ops, busy_ns):
            return {
                "now_ns": now_ns, "accounting_history_complete": True,
                "compute_busy_ns": busy_ns,
                "served_by_category": {"comp": {"int4_ops": ops, "vec_eqops": 0.0}},
                "hbm_served_bytes_by_row_direction": [[0.0, 0.0]],
            }

        previous = None
        envelopes = []
        for index in range(6):
            state = SamplerState.from_counters(
                counters((index + 1) * 10_000_000, (index + 1) * 4.0e11,
                         10_000_000 * (index + 1)))
            window = sampler.window(0, previous, state,
                                    resource_epoch=bridge.resource_epoch)
            previous = state
            result = bridge.ingest(window)
            envelopes.append(result["envelope"])
        self.assertTrue(any(e.domain_fraction("die0") == 0.5 for e in envelopes))


class ThermalCostViewTests(unittest.TestCase):
    def test_scale_and_cache_key_bind_the_epoch(self):
        from joint.thermal_cost import ThermalCostView

        bridge, _ = _make_bridge("closed_loop")
        result = bridge.ingest(_hot_window(bridge, 0, 0, 10_000_000, pe_ops=4.0e10))
        view = ThermalCostView.from_envelope(
            result["envelope"], state_version=bridge.session.state_version,
            hardware_hash=bridge.hardware.content_hash,
            mapping_epoch=bridge.mapping.mapping_epoch)
        self.assertGreaterEqual(view.compute_scale("die0"), 1.0)
        key_a = view.cache_key(horizon_ns=10_000_000)
        self.assertIn(bridge.hardware.content_hash, key_a)
        self.assertIn(view.resource_epoch, key_a)


class ConfigModeTests(unittest.TestCase):
    def test_off_is_default_and_rejects_leftover_config(self):
        from joint.joint_config import JointConfigError, parse_joint_config

        default = parse_joint_config({})
        self.assertEqual(default.thermal_mode, "off")
        self.assertFalse(default.thermal_enabled)
        with self.assertRaisesRegex(JointConfigError, "must be absent"):
            parse_joint_config({"JOINT_THERMAL_CONFIG": "/tmp/does-not-matter.json"})

    def test_non_off_requires_run_config(self):
        from joint.joint_config import JointConfigError, parse_joint_config

        with self.assertRaisesRegex(JointConfigError, "requires JOINT_THERMAL_CONFIG"):
            parse_joint_config({"JOINT_THERMAL_MODE": "closed_loop"})
        resolved = parse_joint_config({
            "JOINT_THERMAL_MODE": "shadow",
            "JOINT_THERMAL_CONFIG": str(LLAMA_DIR / "thermal" / "synthetic_run_config.json"),
        })
        self.assertEqual(resolved.thermal_mode, "shadow")
        self.assertTrue(resolved.thermal_enabled)
        manifest = resolved.manifest_dict()
        self.assertEqual(manifest["thermal_mode"], "shadow")


class ReplayTests(unittest.TestCase):
    def test_written_run_replays_deterministically(self):
        import tempfile

        hardware, mapping, policy = _hardware_mapping_policy()
        installer, _ = _installer()
        bridge = ThermalBridge(run_id="replay-run", config=_run_config("closed_loop"),
                               hardware=hardware, mapping=mapping, policy=policy,
                               installer=installer)
        for seq in range(5):
            bridge.ingest(_hot_window(
                bridge, seq, seq * 10_000_000, (seq + 1) * 10_000_000,
                pe_ops=4.0e10))
        with tempfile.TemporaryDirectory() as directory:
            bridge.write_audit(Path(directory))
            sys.path.insert(0, str(_REPO_ROOT / "sh_test_mesh" / "run_scripts"))
            import thermal_replay

            result = thermal_replay.replay(Path(directory))
        self.assertTrue(result["passed"], result["errors"])
        self.assertEqual(result["windows"], 5)
        self.assertLess(result["max_temperature_delta_k"], 1e-6)
        self.assertEqual(result["committed_time_ns"], 50_000_000)


if __name__ == "__main__":
    unittest.main()
