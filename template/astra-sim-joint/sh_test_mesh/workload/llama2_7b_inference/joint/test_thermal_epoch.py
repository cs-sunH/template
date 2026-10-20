"""Phase D Python-side thermal epoch tests (thermal_activity -> thermal_cap).

Offline only: parsing/validation of the frozen wire, the activity->envelope->cap
translation, the response-assembly attach rule, and safe continue when the
thermal service is off / not ready.  No C++ THERMAL_EPOCH event and no online
run are needed -- those live on the other side of the frozen protocol.
"""
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

_JOINT_FILE = Path(__file__).resolve()
_LLAMA_DIR = _JOINT_FILE.parents[1]
_REPO_ROOT = _JOINT_FILE.parents[4]
_THERMAL_DIR = _REPO_ROOT / "extern" / "thermal_cosim"
for _path in (_THERMAL_DIR,):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from wafercosim.fixtures import (  # noqa: E402
    reference_thermal_config,
    synthetic_thermal_facts_export,
)

from joint.thermal_bridge import (  # noqa: E402
    CoarseCapacityInstaller,
    ThermalBridge,
    ThermalRunConfig,
    build_fixture_mapping,
    hardware_snapshot_from_facts,
    synthetic_rank_to_die,
)
from joint.thermal_epoch import (  # noqa: E402
    RESOURCE_COMPUTE_GATE,
    ThermalEpochError,
    ThermalEpochService,
    envelope_to_thermal_cap,
    parse_thermal_activity,
)
from joint.thermal_policy import ThermalPolicy, ThermalPolicyConfig  # noqa: E402
from online.online_scheduler_base import OnlineSchedulerBase  # noqa: E402

POLICY_PATH = _LLAMA_DIR / "thermal" / "synthetic_policy.json"
THERMAL_DIE_COUNT = 8
WINDOW_NS = 10_000_000
HOT_OPS = 1.2e11


# ---------------------------------------------------------------- fixtures --

def _run_config(mode):
    return ThermalRunConfig(
        mode=mode,
        thermal_config_path="package:thermal/3dice_8die_reference.json",
        energy_table_path="configs/energy/research-v1.json",
        energy_strict=True,
        policy_path=str(POLICY_PATH),
        mapping_epoch="thermal-fixture-v1",
        thermal_step_ns=WINDOW_NS,
        base_frequency_hz=1.0e9,
        accept_assumed_energy=True,
    )


def _shadow_bridge(run_id="epoch-run"):
    hardware = hardware_snapshot_from_facts(synthetic_thermal_facts_export())
    thermal_config = reference_thermal_config()
    mapping = build_fixture_mapping(
        hardware, thermal_config, mapping_epoch="thermal-fixture-v1",
        rank_to_die=synthetic_rank_to_die(54, THERMAL_DIE_COUNT))
    policy = ThermalPolicy(ThermalPolicyConfig.from_json(
        json.loads(POLICY_PATH.read_text())))
    return ThermalBridge(
        run_id=run_id, config=_run_config("shadow"), hardware=hardware,
        mapping=mapping, policy=policy)


def _source(source_id, kind, count):
    return {"source_id": source_id, "action_kind": kind, "count": count}


def _payload(epoch_ns, start_ns, *, sources=(), coverage="measured",
             provenance="coarse_runtime_counters_delta", schema_version=1):
    return {
        "schema_version": schema_version,
        "epoch_ns": epoch_ns,
        "window_start_ns": start_ns,
        "window_end_ns": epoch_ns,
        "coverage": coverage,
        "provenance": provenance,
        "sources": list(sources),
    }


def _parse(payload, **overrides):
    kwargs = dict(run_id="r", hardware_hash="hw", mapping_epoch="mp",
                  resource_epoch=1, activity_seq=0)
    kwargs.update(overrides)
    return parse_thermal_activity(payload, **kwargs)


class _FakeRuntime:
    """Records the capacity updates a closed-loop install makes (no C++)."""

    def __init__(self):
        self.calls = []

    def update_capacity(self, resource, rate, now_ns):
        self.calls.append((int(resource), float(rate), int(now_ns)))


class _DisabledBridge:
    """A bridge that is not enabled (off / not ready)."""

    enabled = False
    run_id = "disabled-run"
    resource_epoch = 1
    hardware = None
    mapping = None


# ------------------------------------------------------- parse / validate --

class ThermalActivityParseTests(unittest.TestCase):
    def test_valid_window_parses_into_an_activity_window(self):
        payload = _payload(10_000_000, 0, sources=[
            _source("die0.comp", "pe", 1000.0),
            _source("die0.comp", "vec", 10.0),
            _source("die0.hbm", "hbm_read", 500.0),
            _source("die0.hbm", "hbm_write", 5.0),
        ])
        window = _parse(payload, run_id="run-1", hardware_hash="h",
                        mapping_epoch="m", resource_epoch=3, activity_seq=7)
        self.assertEqual((window.start_ns, window.end_ns), (0, 10_000_000))
        self.assertEqual(window.run_id, "run-1")
        self.assertEqual(window.hardware_hash, "h")
        self.assertEqual(window.mapping_epoch, "m")
        self.assertEqual(window.resource_epoch, 3)
        self.assertEqual(window.activity_seq, 7)
        self.assertEqual(window.coverage, "measured")
        self.assertEqual(window.provenance, "coarse_runtime_counters_delta")
        actions = {(source, kind): count for source, kind, count in window.actions}
        # the wire's ``comp`` is translated to the mapping's ``compute`` source
        self.assertEqual(actions[("die0.compute", "pe")], 1000.0)
        self.assertEqual(actions[("die0.compute", "vec")], 10.0)
        self.assertEqual(actions[("die0.hbm", "hbm_read")], 500.0)
        self.assertEqual(actions[("die0.hbm", "hbm_write")], 5.0)

    def test_empty_sources_yield_an_explicit_zero_activity_window(self):
        window = _parse(_payload(10_000_000, 0, sources=[]))
        self.assertEqual(window.actions, ())
        self.assertEqual((window.start_ns, window.end_ns), (0, 10_000_000))
        self.assertEqual(window.coverage, "measured")

    def test_coverage_is_passed_through(self):
        window = _parse(_payload(10_000_000, 0, coverage="estimated"))
        self.assertEqual(window.coverage, "estimated")

    def test_unknown_action_kind_is_reported_not_zeroed(self):
        with self.assertRaisesRegex(ThermalEpochError, "unmodelled"):
            _parse(_payload(10_000_000, 0, sources=[
                _source("die0.comp", "noc", 5.0)]))

    def test_window_end_must_equal_epoch(self):
        payload = _payload(10_000_000, 0)
        payload["window_end_ns"] = 5_000_000
        with self.assertRaisesRegex(ThermalEpochError, "epoch_ns"):
            _parse(payload)

    def test_window_start_must_precede_epoch(self):
        with self.assertRaisesRegex(ThermalEpochError, "window_start_ns"):
            _parse(_payload(10_000_000, 10_000_000))

    def test_non_positive_schema_version_is_rejected(self):
        with self.assertRaisesRegex(ThermalEpochError, "schema_version"):
            _parse(_payload(10_000_000, 0, schema_version=2))

    def test_count_must_be_finite_and_non_negative(self):
        with self.assertRaisesRegex(ThermalEpochError, "count"):
            _parse(_payload(10_000_000, 0, sources=[
                _source("die0.comp", "pe", -1.0)]))
        with self.assertRaisesRegex(ThermalEpochError, "count"):
            _parse(_payload(10_000_000, 0, sources=[
                _source("die0.comp", "pe", float("inf"))]))

    def test_action_kind_requires_its_own_source_suffix(self):
        with self.assertRaisesRegex(ThermalEpochError, "source suffix"):
            _parse(_payload(10_000_000, 0, sources=[
                _source("die0.hbm", "pe", 1.0)]))

    def test_source_must_exist_in_the_physical_mapping(self):
        mapping = _shadow_bridge().mapping
        with self.assertRaisesRegex(ThermalEpochError, "physical mapping"):
            _parse(_payload(10_000_000, 0, sources=[
                _source("die99.comp", "pe", 1.0)]), mapping=mapping)

    def test_rank_source_resolves_through_rank_to_die(self):
        # The C++ side emits ``die<rank>``; the fixture mapping is rank % 8, so
        # rank 20 is thermal die 4.
        mapping = _shadow_bridge().mapping
        window = _parse(_payload(10_000_000, 0, sources=[
            _source("die20.comp", "pe", 1.0),
            _source("die20.hbm", "hbm_read", 1.0)]), mapping=mapping)
        self.assertEqual(window.actions[0][0], "die4.compute")
        self.assertEqual(window.actions[1][0], "die4.hbm")

    def test_unresolvable_index_names_the_rejection(self):
        # 99 is neither a mapped rank (ranks 0..53) nor a present thermal die.
        mapping = _shadow_bridge().mapping
        with self.assertRaisesRegex(ThermalEpochError, "neither a mapped rank"):
            _parse(_payload(10_000_000, 0, sources=[
                _source("die99.comp", "pe", 1.0)]), mapping=mapping)


# ----------------------------------------------------------- cap production --

class ThermalCapShapeTests(unittest.TestCase):
    def test_hot_window_produces_the_de_rated_cap_shape(self):
        bridge = _shadow_bridge()
        service = ThermalEpochService(bridge)
        caps = []
        envelopes = []
        for seq in range(6):
            start = seq * WINDOW_NS
            payload = _payload(start + WINDOW_NS, start, sources=[
                _source("die0.comp", "pe", HOT_OPS)])
            caps.append(service.cap_for_request(payload, delivery_sequence=seq))
            envelopes.append(bridge.latest_envelope())
        self.assertTrue(all(cap is not None for cap in caps))
        # the hot fixture drives die0 to the de-rated step (0.5)
        de_rated = [
            (cap, env) for cap, env in zip(caps, envelopes)
            if env.domain_fraction("die0") == 0.5]
        self.assertTrue(de_rated, "hot fixture did not drive die0 to 0.5")
        for cap, envelope in de_rated:
            self.assertEqual(cap["schema_version"], 1)
            self.assertEqual(cap["resource_epoch"], envelope.resource_epoch)
            self.assertEqual(cap["effective_from_ns"], envelope.effective_from_ns)
            self.assertEqual(len(cap["events"]), len(envelope.domains))
            for event, row in zip(cap["events"], envelope.domains):
                self.assertEqual(event["resource_index"], RESOURCE_COMPUTE_GATE)
                self.assertAlmostEqual(
                    event["absolute_physical_rate"],
                    row["compute_throughput_fraction"] * 1.0)
            # shadow must never mark the envelope applied, and the wire flag must
            # say so too: the C++ consumer keys the install decision on it.
            self.assertTrue(all(not row["applied"] for row in envelope.domains))
            self.assertFalse(cap["applied"])

    def test_envelope_to_thermal_cap_is_pure_translation(self):
        bridge = _shadow_bridge()
        service = ThermalEpochService(bridge)
        service.cap_for_request(
            _payload(WINDOW_NS, 0, sources=[_source("die0.comp", "pe", 1.0)]))
        envelope = bridge.latest_envelope()
        cap = envelope_to_thermal_cap(envelope)
        self.assertEqual(cap["events"][0]["resource_index"], RESOURCE_COMPUTE_GATE)
        self.assertEqual(cap["effective_from_ns"], envelope.effective_from_ns)
        self.assertEqual(cap["resource_epoch"], envelope.resource_epoch)

    def test_empty_source_window_advances_and_cools(self):
        bridge = _shadow_bridge()
        service = ThermalEpochService(bridge)

        def die0_max_k():
            rows = bridge.audit_rows("thermal_feedback")
            return rows[-1]["die_block_max_k"]["0"]

        for seq in range(6):
            start = seq * WINDOW_NS
            service.cap_for_request(_payload(
                start + WINDOW_NS, start,
                sources=[_source("die0.comp", "pe", HOT_OPS)]))
        hot_k = die0_max_k()
        self.assertGreater(hot_k, 300.0)
        committed_after_hot = bridge.session.committed_time_ns
        self.assertEqual(committed_after_hot, 6 * WINDOW_NS)

        for idle in range(20):
            start = committed_after_hot + idle * WINDOW_NS
            cap = service.cap_for_request(_payload(
                start + WINDOW_NS, start, sources=[]))
            self.assertIsNotNone(cap)
        self.assertEqual(bridge.session.committed_time_ns,
                         (6 + 20) * WINDOW_NS)
        self.assertLess(die0_max_k(), hot_k, "idle windows must cool the die")


# ------------------------------------------------ response assembly / wiring --

class ThermalCapAssemblyTests(unittest.TestCase):
    def _hook(self, service):
        return SimpleNamespace(thermal_epoch_service=service,
                               thermal_cap_absent_seen=False)

    def test_missing_thermal_activity_never_produces_a_cap(self):
        bridge = _shadow_bridge()
        service = ThermalEpochService(bridge)
        batch = {}
        scheduler = self._hook(service)
        OnlineSchedulerBase._attach_thermal_cap(
            scheduler, {"delivery_sequence": 0}, batch)
        self.assertNotIn("thermal_cap", batch)

    def test_off_mode_produces_no_cap(self):
        batch = {}
        scheduler = self._hook(None)
        delta = {"delivery_sequence": 0, "thermal_activity": _payload(WINDOW_NS, 0)}
        OnlineSchedulerBase._attach_thermal_cap(scheduler, delta, batch)
        self.assertNotIn("thermal_cap", batch)
        self.assertTrue(scheduler.thermal_cap_absent_seen)

    def test_not_ready_bridge_continues_safely(self):
        service = ThermalEpochService(_DisabledBridge())
        batch = {}
        scheduler = self._hook(service)
        delta = {"delivery_sequence": 0,
                 "thermal_activity": _payload(WINDOW_NS, 0)}
        OnlineSchedulerBase._attach_thermal_cap(scheduler, delta, batch)
        self.assertNotIn("thermal_cap", batch)
        self.assertEqual(service.outcome_rows()[0]["outcome"],
                         "thermal_bridge_unavailable")

    def test_activity_bearing_request_gets_the_cap(self):
        bridge = _shadow_bridge()
        service = ThermalEpochService(bridge)
        batch = {}
        scheduler = self._hook(service)
        delta = {"delivery_sequence": 0,
                 "thermal_activity": _payload(WINDOW_NS, 0, sources=[
                     _source("die0.comp", "pe", 1.0)])}
        OnlineSchedulerBase._attach_thermal_cap(scheduler, delta, batch)
        self.assertIn("thermal_cap", batch)
        self.assertEqual(batch["thermal_cap"]["events"][0]["resource_index"],
                         RESOURCE_COMPUTE_GATE)
        self.assertEqual(service.outcome_rows()[0]["outcome"], "cap_produced")


class ClosedLoopEpochTests(unittest.TestCase):
    def test_closed_loop_cap_is_applied_and_bumps_the_epoch(self):
        hardware = hardware_snapshot_from_facts(synthetic_thermal_facts_export())
        mapping = build_fixture_mapping(
            hardware, reference_thermal_config(),
            mapping_epoch="thermal-fixture-v1",
            rank_to_die=synthetic_rank_to_die(54, THERMAL_DIE_COUNT))
        policy = ThermalPolicy(ThermalPolicyConfig.from_json(
            json.loads(POLICY_PATH.read_text())))
        runtime = _FakeRuntime()
        installer = CoarseCapacityInstaller(
            runtime, domain_to_rank={"die0": 0, "die1": 1, "die2": 2, "die3": 3})
        bridge = ThermalBridge(
            run_id="cl-run", config=_run_config("closed_loop"), hardware=hardware,
            mapping=mapping, policy=policy, installer=installer)
        service = ThermalEpochService(bridge)
        before_epoch = bridge.resource_epoch
        cap = service.cap_for_request(
            _payload(WINDOW_NS, 0, sources=[_source("die0.comp", "pe", 1.0)]))
        self.assertIsNotNone(cap)
        self.assertEqual(cap["events"][0]["resource_index"],
                         RESOURCE_COMPUTE_GATE)
        self.assertEqual(runtime.calls[0][0], RESOURCE_COMPUTE_GATE)
        self.assertGreater(bridge.resource_epoch, before_epoch)
        self.assertTrue(all(row["applied"]
                            for row in bridge.latest_envelope().domains))
        self.assertTrue(cap["applied"])
        self.assertTrue(service.outcome_rows()[0]["applied"])


if __name__ == "__main__":
    unittest.main()
