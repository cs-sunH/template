"""Online-service thermal wiring tests (plan V13 / E2).

Proves the online service owns a single run-long thermal session driven by the
JOINT_THERMAL_MODE switch, writes its audit, closes it on the exit path, and
refuses to start a closed_loop run while the C++ activity/install path is absent
(fail-closed, never a faked loop).
"""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

_LLAMA_DIR = Path(__file__).resolve().parents[1]
_REPO_ROOT = _LLAMA_DIR.parents[2]
for _path in (_LLAMA_DIR, _REPO_ROOT / "extern" / "thermal_cosim"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from wafercosim.fixtures import synthetic_resolver_exports  # noqa: E402

import online.online_service as service  # noqa: E402

POLICY_PATH = _LLAMA_DIR / "thermal" / "synthetic_policy.json"


class _StubJointConfig:
    def __init__(self, mode, run_config=None):
        self.thermal_mode = mode
        self.thermal_run_config = run_config
        self.thermal_enabled = mode != "off"


class _StubScheduler:
    def __init__(self, mode, run_config=None):
        self.joint_config = _StubJointConfig(mode, run_config)


class _Args:
    bridge_dir = ""


class ThermalServiceWiringTests(unittest.TestCase):
    def _write_run_config(self, directory: Path, mode: str) -> str:
        hardware, mapping = synthetic_resolver_exports()
        (directory / "resolved_hardware.json").write_text(json.dumps(hardware))
        (directory / "physical_mapping.json").write_text(json.dumps(mapping))
        config = {
            "mode": mode,
            "thermal_config": "package:thermal/3dice_8die_reference.json",
            "energy_table": "configs/energy/research-v1.json",
            "energy_strict": True,
            "policy": str(POLICY_PATH),
            "mapping_epoch": "thermal-fixture-v1",
            "thermal_step_ns": 10_000_000,
            "base_frequency_hz": 1.0e9,
            "accept_assumed_energy": True,
            "hardware_facts": str(directory / "resolved_hardware.json"),
            "mapping": str(directory / "physical_mapping.json"),
        }
        path = directory / "run_config.json"
        path.write_text(json.dumps(config))
        return str(path)

    def test_off_mode_creates_nothing(self):
        args = _Args()
        args.bridge_dir = "/tmp/x"
        bridge, status = service._create_thermal_bridge(
            args, _StubScheduler("off"))
        self.assertIsNone(bridge)
        self.assertEqual(status["thermal_mode"], "off")

    def test_shadow_creates_a_run_long_session_and_writes_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            run_config = self._write_run_config(directory, "shadow")
            args = _Args()
            args.bridge_dir = str(directory / "bridge")
            scheduler = _StubScheduler("shadow", run_config)
            bridge, status = service._create_thermal_bridge(args, scheduler)
            try:
                self.assertIsNotNone(bridge)
                self.assertTrue(bridge.enabled)
                self.assertEqual(status["thermal_mode"], "shadow")
                self.assertEqual(
                    status["thermal_activity_source"],
                    "unavailable_until_cpp_epoch")
                bridge.write_audit()
                audit = Path(status["thermal_audit_dir"])
                self.assertTrue((audit / "thermal_manifest.json").is_file())
                self.assertTrue((audit / "power_windows.jsonl").is_file())
            finally:
                bridge.close()

    def test_closed_loop_refuses_without_a_periodic_thermal_epoch(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            run_config = self._write_run_config(directory, "closed_loop")
            args = _Args()
            args.bridge_dir = str(directory / "bridge")
            with mock.patch.dict(os.environ, {}, clear=False):
                os.environ.pop("SH_THERMAL_EPOCH_NS", None)
                with self.assertRaisesRegex(SystemExit, "SH_THERMAL_EPOCH_NS"):
                    service._create_thermal_bridge(
                        args, _StubScheduler("closed_loop", run_config))

    def test_closed_loop_starts_when_the_thermal_epoch_is_armed(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            run_config = self._write_run_config(directory, "closed_loop")
            args = _Args()
            args.bridge_dir = str(directory / "bridge")
            scheduler = _StubScheduler("closed_loop", run_config)
            with mock.patch.dict(
                    os.environ, {"SH_THERMAL_EPOCH_NS": "100000000"}):
                bridge, status = service._create_thermal_bridge(
                    args, scheduler)
            try:
                self.assertIsNotNone(bridge)
                self.assertTrue(bridge.enabled)
                self.assertEqual(status["thermal_mode"], "closed_loop")
                self.assertEqual(
                    status["thermal_activity_source"],
                    "coarse_runtime_counters_via_thermal_epoch")
                self.assertEqual(status["thermal_epoch_ns"], 100_000_000)
                # Python never installs: the C++ runtime consumes the cap.
                self.assertIsNotNone(bridge.installer)
            finally:
                bridge.close()


if __name__ == "__main__":
    unittest.main()
