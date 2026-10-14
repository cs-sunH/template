"""Contract tests for the external joint acceleration evidence driver."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path

_JOINT_DIR = Path(__file__).resolve().parent
_SCRIPT_DIR = _JOINT_DIR.parents[2] / "run_scripts"
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import joint_accel_acceptance as acceptance  # noqa: E402


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class InventoryTests(unittest.TestCase):
    def test_inventory_is_stable_and_does_not_follow_symlinks(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            tree = base / "tree"
            tree.mkdir()
            (tree / "payload.txt").write_text("stable\n", encoding="utf-8")
            outside = base / "outside.txt"
            outside.write_text("outside\n", encoding="utf-8")
            try:
                (tree / "external-link").symlink_to(outside)
            except OSError as exc:  # pragma: no cover - platform capability
                self.skipTest(f"symlinks unavailable: {exc}")

            entries_a, digest_a = acceptance.inventory_tree(tree)
            entries_b, digest_b = acceptance.inventory_tree(tree)
            by_path = {item["path"]: item for item in entries_a}
            self.assertEqual(digest_a, digest_b)
            self.assertEqual(entries_a, entries_b)
            self.assertEqual(by_path["external-link"]["type"], "symlink")
            self.assertEqual(by_path["external-link"]["link_text"], str(outside))
            self.assertNotIn("outside.txt", {item["path"] for item in entries_a})

    def test_implementation_inventory_excludes_outputs_but_tracks_code_and_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            tree = Path(tmp) / "source"
            (tree / "build/output").mkdir(parents=True)
            (tree / "sh_test_mesh/generated").mkdir(parents=True)
            (tree / "sh_test_mesh/runs/old").mkdir(parents=True)
            (tree / "joint").mkdir()
            code = tree / "joint/kernel.py"
            config = tree / "joint/default.json"
            code.write_text("value = 1\n", encoding="utf-8")
            config.write_text('{"k":1}\n', encoding="utf-8")
            (tree / "README.md").write_text("documentation\n", encoding="utf-8")
            (tree / "build/output/kernel.o").write_bytes(b"build")
            (tree / "sh_test_mesh/generated/result.json").write_text("{}\n", encoding="utf-8")
            (tree / "sh_test_mesh/runs/old/result.json").write_text("{}\n", encoding="utf-8")
            entries, digest = acceptance.implementation_source_inventory(tree)
            paths = {row["path"] for row in entries}
            self.assertIn("joint/kernel.py", paths)
            self.assertIn("joint/default.json", paths)
            self.assertNotIn("README.md", paths)
            self.assertNotIn("build/output/kernel.o", paths)
            self.assertNotIn("sh_test_mesh/generated/result.json", paths)
            self.assertNotIn("sh_test_mesh/runs/old/result.json", paths)
            code.write_text("value = 2\n", encoding="utf-8")
            self.assertNotEqual(digest, acceptance.implementation_source_inventory(tree)[1])

    def test_isolated_source_copy_hashes_then_cleans_only_the_copy(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            (source / "sh_test_mesh/generated/runtime_config").mkdir(parents=True)
            (source / "sh_test_mesh/runs/old").mkdir(parents=True)
            (source / "stable.txt").write_text("preserve\n", encoding="utf-8")
            (source / "sh_test_mesh/generated/runtime_config/config.json").write_text(
                "{}\n", encoding="utf-8")
            (source / "sh_test_mesh/runs/old/result.json").write_text(
                "{}\n", encoding="utf-8")
            copied = root / "attempt/sources/copy"
            record = acceptance._copy_isolated_source(source, copied)
            self.assertEqual(record["source_tree_sha256"],
                             record["copied_source_tree_sha256"])
            self.assertTrue((source / "sh_test_mesh/generated/runtime_config/config.json").is_file())
            self.assertTrue((source / "sh_test_mesh/runs/old/result.json").is_file())
            self.assertFalse((copied / "sh_test_mesh/generated/runtime_config/config.json").exists())
            self.assertFalse((copied / "sh_test_mesh/runs").exists())


class GateIntegrityTests(unittest.TestCase):
    def test_done_requires_matching_result_and_artifact_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gate_dir = root / "gates/G-A0"
            gate_dir.mkdir(parents=True)
            artifact = root / "manifests/input.json"
            artifact.parent.mkdir(parents=True)
            artifact.write_text("{\"input\":1}\n", encoding="utf-8")
            result = gate_dir / "result.json"
            acceptance.write_json(result, {"gate": "G-A0", "status": "PASS"})
            acceptance.write_json(gate_dir / "DONE", {
                "gate": "G-A0", "status": "PASS", "result_sha256": _sha(result),
                "required_artifacts": {"manifests/input.json": _sha(artifact)},
            })
            acceptance.validate_done(root, "G-A0")

            artifact.write_text("{\"input\":2}\n", encoding="utf-8")
            with self.assertRaisesRegex(acceptance.AcceptanceError, "artifact mismatch"):
                acceptance.validate_done(root, "G-A0")

    @staticmethod
    def _seed_a0_a1(root: Path) -> str:
        a0_dir = root / "gates/G-A0"
        a0_dir.mkdir(parents=True)
        a0_result = a0_dir / "result.json"
        acceptance.write_json(a0_result, {"gate": "G-A0", "status": "PASS"})
        acceptance.write_json(a0_dir / "DONE", {
            "gate": "G-A0", "status": "PASS", "result_sha256": _sha(a0_result),
            "parents": {}, "required_artifacts": {},
        })
        a1_dir = root / "gates/G-A1"
        a1_dir.mkdir(parents=True)
        a0_sha = _sha(a0_dir / "DONE")
        a1_result = a1_dir / "result.json"
        acceptance.write_json(a1_result, {"gate": "G-A1", "status": "PASS",
                                          "parents": {"G-A0": a0_sha}})
        acceptance.write_json(a1_dir / "DONE", {
            "gate": "G-A1", "status": "PASS", "result_sha256": _sha(a1_result),
            "parents": {"G-A0": a0_sha}, "required_artifacts": {},
        })
        return _sha(a1_dir / "DONE")

    def test_generic_gate_requires_parent_checks_and_hashed_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a1_sha = self._seed_a0_a1(root)
            proof = root / "manifests/B3-proof.json"
            proof.parent.mkdir(parents=True)
            proof.write_text('{"issue_order":[1,2]}\n', encoding="utf-8")
            gate_dir = root / "gates/G-B3"
            gate_dir.mkdir(parents=True)
            acceptance.write_json(gate_dir / "result.json", {
                "gate": "G-B3", "status": "PASS", "reason_code": "fixture_equal",
                "parents": {"G-A1": a1_sha},
                "checks": {"issue_sequence_equal": True,
                           "callback_ready_next_round": True},
                "required_artifacts": {"manifests/B3-proof.json": _sha(proof)},
            })
            from contextlib import redirect_stdout
            from io import StringIO
            with redirect_stdout(StringIO()):
                code = acceptance.command_gate(root, "G-B3")
            self.assertEqual(code, acceptance.EXIT_PASS)
            done = acceptance.read_json(gate_dir / "DONE")
            self.assertEqual(done["parents"], {"G-A1": a1_sha})
            self.assertEqual(done["required_artifacts"], {
                "manifests/B3-proof.json": _sha(proof)})

            proof.write_text('{"issue_order":[2,1]}\n', encoding="utf-8")
            with redirect_stdout(StringIO()):
                code = acceptance.command_gate(root, "G-B3")
            self.assertEqual(code, acceptance.EXIT_FAIL)
            self.assertFalse((gate_dir / "DONE").exists())
            self.assertEqual(acceptance.read_json(gate_dir / "result.json")["status"],
                             "FAIL")


class FinalChecksTests(unittest.TestCase):
    @staticmethod
    def _write_valid_final_checks(root: Path) -> str:
        (root / "manifests").mkdir(parents=True, exist_ok=True)
        (root / "packages/acceptance").mkdir(parents=True, exist_ok=True)
        entries: list[dict[str, object]] = []
        source_digest = acceptance._canonical_digest(entries)
        source_manifest = root / "manifests/final-source.json"
        acceptance.write_json(source_manifest, {
            "schema_version": acceptance.IMPLEMENTATION_SOURCE_INVENTORY_VERSION,
            "source_root": str(root / "source"),
            "tree_digest_sha256": source_digest, "entries": entries,
        })
        acceptance.write_json(root / "packages/acceptance/baseline.json", {
            "known_baseline_failures": [],
        })
        command_specs = [
            ("build_all", ["cmake", "--build", "build/all", "-j4"], False),
            ("ctest_registered", ["ctest", "--test-dir", "build/all",
                                  "--output-on-failure"], True),
            ("pytest_related", ["python3", "-m", "pytest", "joint"], True),
            ("unittest_slo_tools", ["python3", "-m", "unittest", "discover"], True),
            ("cpp_direct", ["build/all/direct-fixture"], True),
        ]
        commands = []
        for index, (kind, argv, is_test) in enumerate(command_specs):
            log_path = root / f"logs/{kind}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text(f"{kind} passed\n", encoding="utf-8")
            command = {
                "kind": kind, "argv": argv, "cwd": str(root),
                "status": "completed", "returncode": 0, "completed": True,
                "log": str(log_path.relative_to(root)),
                "log_sha256": _sha(log_path),
            }
            if is_test:
                command["tests_run"] = 1
            commands.append(command)
        artifact_paths = [
            "native/libjoint_projection.so",
            "native/extern/die_model/libdie_model_cabi.so",
            "build/all/CMakeCache.txt",
            "source/sh_test_mesh/workload/llama2_7b_inference/joint/prediction_config.py",
        ]
        artifacts = []
        for relative in artifact_paths:
            artifact = root / relative
            artifact.parent.mkdir(parents=True, exist_ok=True)
            artifact.write_text(relative + "\n", encoding="utf-8")
            artifacts.append({"path": relative, "sha256": _sha(artifact)})
        acceptance.write_json(root / "packages/acceptance/final-checks.json", {
            "schema_version": "joint-accel-final-checks-v1-20261006",
            "status": "PASS", "source_tree_digest_sha256": source_digest,
            "source_manifest": {"path": "manifests/final-source.json",
                                 "sha256": _sha(source_manifest)},
            "commands": commands, "artifacts": artifacts,
            "baseline_failures": [], "new_failures": [],
        })
        return source_digest

    def test_final_checks_accepts_hashed_full_suite_and_required_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            digest = self._write_valid_final_checks(root)
            status, reason, _details = acceptance._final_checks_status(root, digest)
            self.assertEqual((status, reason), ("PASS", "final_checks_verified"))

    def test_final_checks_rejects_a_targeted_build_mislabeled_as_all(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            digest = self._write_valid_final_checks(root)
            checks_path = root / "packages/acceptance/final-checks.json"
            checks = acceptance.read_json(checks_path)
            checks["commands"][0]["argv"] = [
                "cmake", "--build", "build/all", "--target", "AstraSim_Analytical",
            ]
            acceptance.write_json(checks_path, checks)
            status, reason, _details = acceptance._final_checks_status(root, digest)
            self.assertEqual((status, reason), ("FAIL", "final_build_not_all_target"))

    def test_final_checks_rejects_filtered_ctest_and_unbound_source_entries(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            digest = self._write_valid_final_checks(root)
            checks_path = root / "packages/acceptance/final-checks.json"
            checks = acceptance.read_json(checks_path)
            checks["commands"][1]["argv"].extend(["-R", "Window"])
            acceptance.write_json(checks_path, checks)
            status, reason, _details = acceptance._final_checks_status(root, digest)
            self.assertEqual((status, reason),
                             ("FAIL", "final_ctest_not_full_registered_suite"))

            checks["commands"][1]["argv"] = [
                "ctest", "--test-dir", "build/all", "--output-on-failure"]
            source_manifest = root / "manifests/final-source.json"
            source_wire = acceptance.read_json(source_manifest)
            source_wire["entries"] = [{"path": "src/fake.cc", "sha256": "f" * 64}]
            acceptance.write_json(source_manifest, source_wire)
            checks["source_manifest"]["sha256"] = _sha(source_manifest)
            acceptance.write_json(checks_path, checks)
            status, reason, _details = acceptance._final_checks_status(root, digest)
            self.assertEqual((status, reason),
                             ("FAIL", "final_source_manifest_content_mismatch"))

    def test_f2_keeps_science_status_separate_from_delivery_readiness(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "gates/G-F0").mkdir(parents=True)
            (root / "gates/G-F1").mkdir(parents=True)
            for relative in ("gates/G-F0/result.json", "gates/G-F1/result.json",
                             "packages/acceptance/verify-clean.json",
                             "packages/acceptance/final-checks.json",
                             "packages/acceptance/report-source.json",
                             "manifests/implementation-source.json"):
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("{}\n", encoding="utf-8")
            statuses = {
                "G-F0": {"status": "FAIL", "reason_code": "heldout_regression",
                         "result_sha256": "a" * 64},
                "G-F1": {"status": "INSUFFICIENT_EVIDENCE",
                         "reason_code": "too_few_pairs", "result_sha256": "b" * 64},
                "G-E0": {"status": "PASS", "reason_code": "fixture_pass",
                         "result_sha256": "c" * 64, "done_sha256": "d" * 64},
            }
            clean = {
                "status": "PASS",
                "checks": {"current_source_matches_captured_source": True,
                           "release_window_policy_passed": True},
                "implementation_source": {
                    "tree_digest_sha256": "e" * 64,
                    "inventory_path": "manifests/implementation-source.json",
                },
            }
            with mock.patch.object(acceptance, "gate_status_detail",
                                   side_effect=lambda _root, name: statuses[name]), \
                    mock.patch.object(acceptance, "_final_checks_status",
                                      return_value=("PASS", "final_checks_verified", {
                                          "source_tree_digest_sha256": "e" * 64,
                                          "commands": [], "artifacts": []})):
                acceptance._publish_f2_gate_result(root, clean)
            result = acceptance.read_json(root / "gates/G-F2/result.json")
            self.assertEqual(result["status"], "PASS")
            self.assertEqual(result["parents"], {"G-E0": "d" * 64})
            self.assertEqual(result["science_status"]["G-F0"]["status"], "FAIL")
            self.assertEqual(result["science_status"]["G-F1"]["status"],
                             "INSUFFICIENT_EVIDENCE")

    def test_generic_gate_preserves_insufficient_evidence_without_done(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gate_dir = root / "gates/G-F0"
            gate_dir.mkdir(parents=True)
            acceptance.write_json(gate_dir / "result.json", {
                "gate": "G-F0", "status": "INSUFFICIENT_EVIDENCE",
                "reason_code": "holdout_cluster_count_below_threshold"})
            from contextlib import redirect_stdout
            from io import StringIO
            with redirect_stdout(StringIO()):
                code = acceptance.command_gate(root, "G-F0")
            self.assertEqual(code, acceptance.EXIT_INSUFFICIENT)
            self.assertFalse((gate_dir / "DONE").exists())

    def test_a1_fixture_requires_two_identical_expected_outputs(self):
        expected = acceptance.NUMERIC_EXPECTED
        runs = [
            {"status": "completed", "actual": dict(expected)},
            {"status": "completed", "actual": dict(expected)},
        ]
        self.assertTrue(acceptance.numeric_fixture_repeated(runs))
        runs[1]["actual"]["predicted_settle_done_ns"] += 1
        self.assertFalse(acceptance.numeric_fixture_repeated(runs))
        self.assertFalse(acceptance.numeric_fixture_repeated(runs[:1]))

    def test_nonzero_pytest_is_baseline_failure_only_with_named_case(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "pytest.log"
            log.write_text(
                "E AssertionError: expected 4, got 3\n"
                "FAILED tests/test_example.py::test_case[param-a] - AssertionError\n",
                encoding="utf-8")
            captured = acceptance.classify_test_command({
                "argv": ["python3", "-m", "pytest", "-q"],
                "status": "nonzero_exit", "returncode": 1,
                "log": str(log), "log_sha256": _sha(log),
            })
            self.assertEqual(captured["evidence_status"], "captured_failure")
            failure = captured["failure_cases"][0]
            self.assertEqual(failure["test_name"], "tests/test_example.py::test_case[param-a]")
            self.assertTrue(failure["input_signature"])
            self.assertTrue(failure["error_signature"])

            log.write_text("ERROR collecting tests/test_example.py\n", encoding="utf-8")
            incomplete = acceptance.classify_test_command({
                "argv": ["python3", "-m", "pytest", "-q"],
                "status": "nonzero_exit", "returncode": 1,
                "log": str(log), "log_sha256": _sha(log),
            })
            self.assertEqual(incomplete["evidence_status"], "incomplete")

    def test_success_exit_with_missing_log_is_incomplete(self):
        result = acceptance.classify_test_command({
            "argv": ["python3", "-m", "pytest"], "status": "completed",
            "returncode": 0, "log": "/missing/test.log"})
        self.assertEqual(result["evidence_status"], "incomplete")

    def test_pytest_error_signatures_are_scoped_to_each_failure_block(self):
        output = (
            "___ A.test_first ___\nE ValueError: first failure\n"
            "___ B.test_second ___\nE RuntimeError: second failure\n"
            "=========================== short test summary info ===========================\n"
            "FAILED path/test.py::A::test_first\n"
            "FAILED path/test.py::B::test_second\n")
        failures = acceptance._failure_cases(["python3", "-m", "pytest"], output)
        self.assertEqual(len(failures), 2)
        self.assertIn("ValueError: first failure", failures[0]["error_excerpt"])
        self.assertNotIn("second failure", failures[0]["error_excerpt"])
        self.assertIn("RuntimeError: second failure", failures[1]["error_excerpt"])
        self.assertNotEqual(failures[0]["error_signature"], failures[1]["error_signature"])


class MissingEvidenceTests(unittest.TestCase):
    def test_performance_cli_dispatches_exact_diagnostic_without_unused_native_flag(self):
        parsed = acceptance.build_parser().parse_args([
            "performance", "--evidence", "/tmp/evidence", "--exact-only"])
        self.assertTrue(parsed.exact_only)
        from contextlib import redirect_stderr
        from io import StringIO
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            acceptance.build_parser().parse_args([
                "performance", "--evidence", "/tmp/evidence",
                "--native-library", "/tmp/libjoint_projection.so"])

    def test_smoke_without_built_source_copy_is_incomplete(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "reference/source").mkdir(parents=True)
            (root / "manifests").mkdir()
            from contextlib import redirect_stdout
            from io import StringIO

            stdout = StringIO()
            with redirect_stdout(stdout):
                result = acceptance.main([
                    "smoke", "--evidence", str(root), "--variant", "exact"])
            self.assertEqual(result, acceptance.EXIT_INCOMPLETE)
            payload = json.loads(stdout.getvalue())
            self.assertEqual(payload["status"], "INCOMPLETE")
            self.assertEqual(payload["reason_code"], "smoke_source_copy_required")
            self.assertFalse((root / "gates/G-A1/DONE").exists())

    def test_smoke_uses_documented_manual_exact_copy_when_source_is_omitted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "reference/source").mkdir(parents=True)
            (root / "manifests").mkdir()
            manual = root / "runs/manual-exact/source"
            manual.mkdir(parents=True)
            from contextlib import redirect_stdout
            from io import StringIO

            stdout = StringIO()
            with redirect_stdout(stdout):
                code = acceptance.main([
                    "smoke", "--evidence", str(root), "--variant", "exact"])
            self.assertEqual(code, acceptance.EXIT_INCOMPLETE)
            result = acceptance.read_json(root / "packages/acceptance/smoke.json")
            self.assertEqual(result["source_resolution"]["selected_by"], "manual_exact")
            self.assertEqual(result["reason_code"], "smoke_parent_gates_missing")
            self.assertFalse((root / "runs/smoke/attempt-001/sources/source").exists())

    def test_performance_stops_before_launch_when_parent_gates_are_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "reference/source").mkdir(parents=True)
            (root / "manifests").mkdir()
            from contextlib import redirect_stdout
            from io import StringIO

            with redirect_stdout(StringIO()):
                code = acceptance.main(["performance", "--evidence", str(root)])
            self.assertEqual(code, acceptance.EXIT_INCOMPLETE)
            result = acceptance.read_json(root / "packages/acceptance/performance.json")
            self.assertEqual(result["reason_code"], "performance_parent_gates_missing")
            self.assertEqual(result["repetitions"], [])
            self.assertEqual(result["invocations"], [])

    def test_exact_only_performance_diagnostic_does_not_require_accuracy_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "reference/source").mkdir(parents=True)
            (root / "manifests").mkdir()
            from contextlib import redirect_stdout
            from io import StringIO

            with redirect_stdout(StringIO()):
                code = acceptance.main([
                    "performance", "--evidence", str(root), "--exact-only"])
            self.assertEqual(code, acceptance.EXIT_INCOMPLETE)
            result = acceptance.read_json(root / "packages/acceptance/performance.json")
            self.assertEqual(result["mode"], "exact_only_diagnostic")
            self.assertFalse(result["formal_speed_claim"])
            self.assertEqual(result["reason_code"], "performance_parent_gates_missing")
            self.assertEqual(result["missing_parent_gates"], ["G-C5", "G-D1", "G-E0"])
            self.assertEqual(result["invocations"], [])

    def test_report_keeps_implementation_accuracy_research_and_speed_separate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "reference/source").mkdir(parents=True)
            (root / "manifests").mkdir()
            from contextlib import redirect_stdout
            from io import StringIO

            with redirect_stdout(StringIO()):
                code = acceptance.command_report(root)
            self.assertEqual(code, acceptance.EXIT_INCOMPLETE)
            report = acceptance.read_json(root / "packages/acceptance/report.json")
            self.assertFalse(report["implementation_complete"])
            self.assertFalse(report["accuracy_accepted"])
            self.assertFalse(report["research_conclusions_validated"])
            self.assertIsNone(report["speed_goal_met"])
            # 无冻结验收记录时发布一致性不可判定（2026-10-09 起发布
            # 默认 = F 链验收档 K2/ppm0，仅在有 frozen-accuracy 时比对）。
            self.assertEqual(report["release_window_policy"]["status"],
                             "INCOMPLETE")
            self.assertEqual(report["release_window_policy"]["reason_code"],
                             "frozen_accuracy_missing")
            self.assertTrue((root / "packages/acceptance/acceptance.md").is_file())
            self.assertTrue((root / "packages/acceptance/failure-samples.json").is_file())

    def test_report_preserves_fail_and_insufficient_gate_states(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "reference/source").mkdir(parents=True)
            (root / "manifests").mkdir()
            fail_dir = root / "gates/G-B3"
            fail_dir.mkdir(parents=True)
            acceptance.write_json(fail_dir / "result.json", {
                "gate": "G-B3", "status": "FAIL", "reason_code": "sequence_changed"})
            insufficient_dir = root / "gates/G-F0"
            insufficient_dir.mkdir(parents=True)
            acceptance.write_json(insufficient_dir / "result.json", {
                "gate": "G-F0", "status": "INSUFFICIENT_EVIDENCE",
                "reason_code": "too_few_clusters"})
            from contextlib import redirect_stdout
            from io import StringIO
            with redirect_stdout(StringIO()):
                code = acceptance.command_report(root)
            report = acceptance.read_json(root / "packages/acceptance/acceptance.json")
            self.assertEqual(code, acceptance.EXIT_FAIL)
            self.assertEqual(report["gates"]["G-B3"]["status"], "FAIL")
            self.assertEqual(report["gates"]["G-F0"]["status"],
                             "INSUFFICIENT_EVIDENCE")
            self.assertTrue((root / "gates/G-F1/result.json").is_file())
            self.assertEqual(acceptance.read_json(
                root / "packages/acceptance/report.json"), report)

    def test_verify_clean_is_read_only_and_checks_saved_user_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "evidence"
            workspace = Path(tmp) / "workspace"
            (root / "reference/source").mkdir(parents=True)
            (root / "manifests").mkdir()
            trace_config = workspace / "sh_test_mesh/workload/llama2_7b_inference/trace_config.csv"
            trace_config.parent.mkdir(parents=True)
            trace_config.write_bytes(b"original placeholder\n")
            traces = trace_config.parent / "traces"
            traces.mkdir()
            (traces / "materializer.py").write_text("pass\n", encoding="utf-8")
            prediction_module = (workspace / "sh_test_mesh/workload/llama2_7b_inference"
                                / "joint/prediction_config.py")
            prediction_module.parent.mkdir(parents=True)
            prediction_module.write_text(
                "class PredictionParams:\n"
                "    window_tile_limit = 1\n"
                "    window_local_error_ppm = 10000\n"
                "    window_refine_margin_ppm = 20000\n", encoding="utf-8")
            acceptance.write_json(root / "manifests/preexisting-generated-artifacts.json",
                                  {"entries": []})
            (root / "manifests/trace_config.original").write_bytes(
                trace_config.read_bytes())
            acceptance.write_json(root / "manifests/source-inventory.json",
                                  {"entries": []})
            old_root = acceptance.REPO_ROOT
            acceptance.REPO_ROOT = workspace
            try:
                binding = acceptance._save_implementation_source_binding(root, workspace)
                acceptance.write_json(root / "packages/acceptance/capture.json", {
                    "status": "INCOMPLETE", "source_root": str(workspace),
                    "implementation_source": binding,
                })
                from contextlib import redirect_stdout
                from io import StringIO
                with redirect_stdout(StringIO()):
                    code = acceptance.command_verify_clean(root)
            finally:
                acceptance.REPO_ROOT = old_root
            # 无 frozen-accuracy 的临时 root：发布一致性不可判定
            # （policy INCOMPLETE/frozen_accuracy_missing），verify-clean
            # 整体 INCOMPLETE（交付检查不完备）；源绑定等其余分量照常。
            self.assertEqual(code, acceptance.EXIT_INCOMPLETE)
            clean = acceptance.read_json(root / "packages/acceptance/verify-clean.json")
            self.assertTrue(clean["checks"]["current_source_matches_captured_source"])
            self.assertEqual(clean["release_window_policy"]["status"],
                             "INCOMPLETE")
            self.assertEqual(clean["release_window_policy"].get("reason_code"),
                             "frozen_accuracy_missing")
            self.assertTrue(trace_config.is_file())
            self.assertEqual(trace_config.read_bytes(), b"original placeholder\n")


class ColdFixtureContractTests(unittest.TestCase):
    @staticmethod
    def _empty_initial_state():
        return {
            "tick_ns": 0, "next_flow_id": 0, "next_tag": 1,
            "ranks": [{"rank": 0, "next_store_id": 1}],
            "components": [{
                "identity": {"kind": "rank", "rank": 0}, "version": 0,
                "state": {
                    "nodes": [], "ready": [], "deferred": [], "events": [],
                    "flows": [], "propagation_tails": [], "hbm_jobs": [],
                    "pool_jobs": [], "watches": [], "mailboxes": [],
                    "descriptors": [], "cursors": [], "buffers": [],
                    "obligations": [], "hardware_inflight": 0,
                },
            }],
            "directory": {"entries": []}, "capacity": {"reservations": []},
            "history": {"events": []},
        }

    def _bundle(self, tmp: Path):
        asset = tmp / "system.json"
        asset.write_text("{}\n", encoding="utf-8")
        runtime_asset = tmp / "python-runtime.json"
        acceptance.write_json(runtime_asset, {
            "version": acceptance.PYTHON_RUNTIME_ASSET_VERSION,
            "face_trace": {
                "model": {"layers": 2, "hidden_size": 8, "ffn_size": 16,
                          "num_heads": 2, "vocab_size": 32,
                          "bytes_per_elem": 2, "mlp_variant": "gelu"},
                "model_name": "cold-fixture", "position_encoding": "none",
                "remote_operand_loads": False,
                "kv_reserve_context_tokens": 0, "prefill_chunk_size": 8,
                "trace_granularity": "token_expanded",
                "hardware_capacity_profile": "cold-fixture",
            },
            "hardware": {"mesh_rows": 1, "mesh_cols": 1,
                          "local_hbm_capacity_bytes": 1048576,
                          "local_hbm_bandwidth_gbps": 100.0,
                          "d2d_bandwidth_gbps": 50.0,
                          "peak_perf_tflops": 1.0,
                          "d2d_latency_ns": 0, "local_hbm_latency_ns": 0,
                          "label": "cold-fixture"},
            "remote_memory": {"edge_npus": [0], "mesh_rows": 1, "mesh_cols": 1,
                               "remote_mem_latency_ns": 0,
                               "remote_mem_bw_gbps": 100.0,
                               "logical_pool": "cold-fixture-pool"},
            "inference_groups": [{"name": "instance-0", "pg_name": "pg0",
                                  "ranks": [0]}],
            "joint_config": {
                "category_mode": "lru", "scheduler_mode": "joint",
                "layer_policy": "minimal_layer_groups",
                "allowed_actions": "all", "quota_mode": "off",
                "boundary_policy": "off", "combo": None,
            },
            "system_backend": {"compute_cost_model": "norm", "die_profile": None,
                               "die_model_config_asset_id": None},
        })
        state = {
            "version": acceptance.FIXTURE_STATE_VERSION, "kind": "cold_initial",
            "case_id": "M03-seed000", "cluster_id": "M03/seed000",
            "scenario_id": "M03", "seed": 0, "request_id": "r0",
            "fixed_instance_id": "decode-1", "fixed_action": "copy",
            "causal_input": {"focus_stage": "D", "context_tokens": 128,
                              "horizon_tokens": 1, "competition_level": "low",
                              "source_count": 1, "committed_decode_tokens": 2},
            "assets": [{"asset_id": "sys", "path": "system.json",
                        "sha256": acceptance.sha256_file(asset)},
                       {"asset_id": "runtime", "path": "python-runtime.json",
                        "sha256": acceptance.sha256_file(runtime_asset)}],
            "python_runtime_asset_id": "runtime",
            "hardware": {"rank_count": 1, "system_asset_id": "sys",
                         "network_asset_id": "sys", "remote_memory_asset_id": None,
                         "die_hardware_asset_id": None},
            "initial_state": self._empty_initial_state(),
        }
        contract = {
            "projection_abi_version": 1, "descriptor_version": 1,
            "pipeline_protocol_version": 1,
            "model_version": "v7-three-actions-20261014",
            "timer_whitelist_version": "timer-whitelist-v1-20261005",
            "window_tile_limit": 1, "window_local_error_ppm": 10000,
            "window_refine_margin_ppm": 20000,
            "prediction_config_sha256": "a" * 64,
        }
        result = {
            "version": acceptance.FIXTURE_RESULT_VERSION,
            "case_id": state["case_id"], "cluster_id": state["cluster_id"],
            "scenario_id": state["scenario_id"], "seed": 0,
            "request_id": "r0", "fixed_instance_id": "decode-1",
            "fixed_action": "copy", "kernel_contract": dict(contract),
            "status": "PASS", "reason": "completed", "completed": True,
            "initial_certificate": self._empty_initial_state(),
            "terminal_certificate": {"tick_ns": 5, "settled": True},
            "measurements": [
                {"stage": "P", "fixed_instance_id": "prefill-0",
                 "fixed_action": "copy", "boundary_ordinal": 0,
                 "start_tick_ns": 0, "service_tick_ns": 2, "settle_tick_ns": 3,
                 "logical_event_count": 2, "native_physical_steps": 2,
                 "window_stats": {"effective_k": 1}, "charged_totals": {"bytes": 10}},
                {"stage": "D", "fixed_instance_id": "decode-1",
                 "fixed_action": "copy", "boundary_ordinal": 0,
                 "start_tick_ns": 3, "service_tick_ns": 4, "settle_tick_ns": 5,
                 "logical_event_count": 2, "native_physical_steps": 2,
                 "window_stats": {"effective_k": 1}, "charged_totals": {"bytes": 20}},
            ],
            "boundary_records": [],
            "audit_records": [],
        }
        for stage, ordinal, tick, instance, action in (
                ("P", 0, 0, "prefill-0", "copy"),
                ("D", 0, 3, "decode-1", "copy")):
            if stage == "P":
                session_view = {key: None for key in (
                                   "session_id", "home_instance", "resident_instance",
                                   "location", "history_tokens", "resident_prefix_layers",
                                   "history_bytes_by_tp_rank", "missing_bytes_by_tp_rank")}
                request_view = {key: None for key in (
                                   "request_id", "session_id", "input_tokens",
                                   "history_tokens_before", "estimated_decode_tokens",
                                   "horizon_source",
                                   "input_kv_bytes_by_tp_rank")}
                session_view["session_id"] = request_view["session_id"] = "s0"
                request_view["request_id"] = "r0"
                stage_input = {"kind": "P", "session_view": session_view,
                               "request_view": request_view,
                               "prep_frontier_by_instance": {}}
            else:
                stage_input = {
                    "kind": "D",
                    "boundary_view": {key: None for key in (
                        "session_id", "home_instance", "p_instance_index",
                        "p_action", "history_tokens",
                        "working_bytes_by_tp_rank", "decode_steps",
                        "kv_token_bytes", "logic_home_instance",
                        "kv_token_bytes_by_tp_rank")},
                    "runtime_state": {
                        "terminated": False,
                        "first_token": {"issued": False, "consumed": False,
                                        "rng_advances": 0,
                                        "rebuild_sampling_suppressed": False},
                        "obligations": [], "working_layers": []},
                               "decision_epoch": 0,
                               "inherited_obligation_ids": []}
            forecast_inputs = {"context_tokens": 128, "horizon_tokens": 1,
                               "generator_inputs": {}}
            forecast_identity = "no-exogenous-generator/v7-three-actions-20261014"
            planner_state = {
                "version": acceptance.PLANNER_STATE_VERSION,
                "stage_input": stage_input,
                "instance_loads": [{
                    "instance_index": 0, "queued_task_load_ns": 0,
                    "running_task_load_ns": 0, "active_decode_task_load_ns": 0,
                    "hbm_remaining_bytes_by_tp_rank": {},
                    "reclaimable_bytes_by_tp_rank": {}, "capacity_summary": None,
                    "capacity_ledger_by_tp_rank": {},
                }],
                "registries": {
                    "service_factors": {"prefill_factor": 1.0, "decode_factor": 1.0,
                                         "transfer_factor": 1.0, "updates": {},
                                         "rejected": {}, "anomalous": {},
                                         "contended": {}},
                    "noc_flows": [], "pool_edge_counts": [], "hbm_port_counts": [],
                    "telemetry": {"rates": {}, "flow_counts": {}, "window_ns": 0},
                    "forecast_history": [],
                },
                "forecast_generator": {
                    "identity": forecast_identity,
                    "causal_inputs": forecast_inputs,
                    "semantic_sha256": acceptance._canonical_digest({
                        "identity": forecast_identity,
                        "causal_inputs": forecast_inputs}),
                },
            }
            result["boundary_records"].append({
                "request_id": "r0", "stage": stage, "boundary_ordinal": ordinal,
                "tick_ns": tick, "fixed_instance_id": instance, "fixed_action": action,
                "causal_prestate": {"work_ledger": {}, "resources": {},
                                    "directory": {}, "capacity": {},
                                    "obligations": {}, "history": {}},
                "causal_metadata": {"context_tokens": 128, "horizon_tokens": 1},
                "planner_state": planner_state,
            })
        return state, contract, result

    def test_exact_contract_and_cold_result_keep_stage_local_actions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state, contract, result = self._bundle(root)
            state_path = root / "state.json"
            state_path.write_text("{}", encoding="utf-8")
            parsed = acceptance.validate_fixture_state(state, state_path)
            actual_contract = acceptance.validate_kernel_contract(contract)
            accepted = acceptance.validate_fixture_result(result, parsed, actual_contract)
            self.assertEqual(accepted["measurements"][0]["fixed_action"], "copy")
            self.assertEqual(accepted["measurements"][1]["fixed_action"], "copy")

    def test_cold_initial_state_allows_loaded_logical_directory_capacity_and_history(self):
        initial = self._empty_initial_state()
        initial["directory"] = {"entries": [{"session_id": "s0", "home": 0}]}
        initial["capacity"] = {"capacity_by_rank": [1024], "reservations": []}
        initial["history"] = {"version": 1, "run": {"count": 0}, "events": []}
        parsed = acceptance.validate_cold_initial_state(initial, rank_count=1)
        self.assertEqual(parsed["directory"]["entries"][0]["session_id"], "s0")

    def test_prediction_boundary_cannot_contain_execution_decode_length(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state, contract, result = self._bundle(root)
            state_path = root / "state.json"
            state_path.write_text("{}", encoding="utf-8")
            parsed = acceptance.validate_fixture_state(state, state_path)
            result["boundary_records"][0]["causal_metadata"]["committed_decode_tokens"] = 2
            with self.assertRaisesRegex(acceptance.AcceptanceError,
                                        "leaks committed_decode_tokens"):
                acceptance.validate_fixture_result(result, parsed, contract)

    def test_fixture_requires_verified_python_runtime_asset_and_stage_planner(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state, contract, result = self._bundle(root)
            state_path = root / "state.json"
            state_path.write_text("{}", encoding="utf-8")
            missing_runtime = dict(state)
            missing_runtime.pop("python_runtime_asset_id")
            with self.assertRaisesRegex(acceptance.AcceptanceError,
                                        "fixture state keys mismatch"):
                acceptance.validate_fixture_state(missing_runtime, state_path)
            bad_planner = dict(result)
            bad_planner["boundary_records"] = [dict(row) for row in result["boundary_records"]]
            bad_planner["boundary_records"][0]["planner_state"] = dict(
                bad_planner["boundary_records"][0]["planner_state"])
            bad_planner["boundary_records"][0]["planner_state"]["version"] = "wrong"
            parsed_state = acceptance.validate_fixture_state(state, state_path)
            with self.assertRaisesRegex(acceptance.AcceptanceError,
                                        "planner_state.version mismatch"):
                acceptance.validate_fixture_result(bad_planner, parsed_state, contract)
            wrong_load_key = json.loads(json.dumps(result))
            load = wrong_load_key["boundary_records"][0]["planner_state"]["instance_loads"][0]
            load["reclaimable_bytes_by_rank"] = load.pop("reclaimable_bytes_by_tp_rank")
            with self.assertRaisesRegex(acceptance.AcceptanceError,
                                        "planner_state.instance_loads\\[0\\].*keys mismatch"):
                acceptance.validate_fixture_result(wrong_load_key, parsed_state, contract)

    def test_event_tape_uses_strict_stable_ordinals_and_real_trigger_shape(self):
        tape = {"version": acceptance.FIXTURE_TAPE_VERSION, "case_id": "x",
                "events": [
                    {"ordinal": 0, "trigger": {"kind": "tick", "tick_ns": 4},
                     "kind": "graph_commit", "payload": {
                         "batch": {"schema_version": 3},
                         "decision_events": [],
                         "boundary_record": {"stage": "P"}}},
                    {"ordinal": 1, "trigger": {"kind": "gate_terminal",
                                                "segment_id": "seg", "role": "service"},
                     "kind": "external_resource", "payload": {
                         "resource": "noc_link", "mutation": "join",
                         "target": {"src_rank": 0, "dst_rank": 1},
                         "amount": {"bytes": 1024, "tag": 3}}},
                ]}
        self.assertEqual(acceptance.validate_event_tape(tape, case_id="x"), tape)
        closed = json.loads(json.dumps(tape))
        closed["events"][1]["kind"] = "resource_change"
        with self.assertRaisesRegex(acceptance.AcceptanceError,
                                    "outside the closed exogenous API"):
            acceptance.validate_event_tape(closed, case_id="x")
        unsupported = json.loads(json.dumps(tape))
        unsupported["events"][1]["payload"]["mutation"] = "leave"
        with self.assertRaisesRegex(acceptance.AcceptanceError,
                                    "no real backend API"):
            acceptance.validate_event_tape(unsupported, case_id="x")
        bad_batch = json.loads(json.dumps(tape))
        bad_batch["events"][0]["payload"]["batch"] = {"schema_version": 1}
        with self.assertRaisesRegex(acceptance.AcceptanceError,
                                    "actual schema-3 batch"):
            acceptance.validate_event_tape(bad_batch, case_id="x")
        tape["events"][1]["ordinal"] = 0
        with self.assertRaisesRegex(acceptance.AcceptanceError, "strict stable order"):
            acceptance.validate_event_tape(tape, case_id="x")

    def test_raw_fixture_result_keeps_canonicalization_pending(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state, contract, result = self._bundle(root)
            state_path = root / "state.json"
            state_path.write_text("{}", encoding="utf-8")
            parsed_state = acceptance.validate_fixture_state(state, state_path)
            raw = json.loads(json.dumps(result))
            raw["canonicalization_pending"] = True
            for row in raw["boundary_records"]:
                row["causal_prestate"] = None
                row["planner_state"] = None
            raw["audit_records"] = [
                {"kind": acceptance.RAW_FRAME_AUDIT_KIND, "boundary_ordinal": 0,
                 "frame": {"schema_version": 4, "protocol_version": 4,
                           "bridge_protocol": "bridge-v4-20261011",
                           "work_ledger_schema": "workledger-v4-20261011",
                           "producer": "cpp_online"},
                 "runtime_snapshot": {"active_resources": []},
                 "producer_build": {"binary_sha256": "b" * 64,
                                    "kernel_contract": dict(contract)}},
                {"kind": acceptance.RAW_FRAME_AUDIT_KIND, "boundary_ordinal": 0,
                 "frame": {"schema_version": 4, "protocol_version": 4,
                           "bridge_protocol": "bridge-v4-20261011",
                           "work_ledger_schema": "workledger-v4-20261011",
                           "producer": "cpp_online"},
                 "runtime_snapshot": {"active_resources": []},
                 "producer_build": {"binary_sha256": "b" * 64,
                                    "kernel_contract": dict(contract)}},
            ]
            accepted = acceptance.validate_raw_fixture_result(raw, parsed_state, contract)
            self.assertTrue(accepted["canonicalization_pending"])
            # Preserve the exact V3 archive route independently of the
            # current V4 positive fixture; both require complete identities.
            archive_v3 = json.loads(json.dumps(raw))
            for entry in archive_v3["audit_records"]:
                entry["frame"].update({
                    "schema_version": 3,
                    "protocol_version": 3,
                    "bridge_protocol": "bridge-v3-20261009",
                    "work_ledger_schema": "workledger-v3-20261009",
                })
            archived = acceptance.validate_raw_fixture_result(
                archive_v3, parsed_state, contract)
            self.assertTrue(archived["canonicalization_pending"])
            not_pending = json.loads(json.dumps(raw))
            not_pending["canonicalization_pending"] = False
            with self.assertRaisesRegex(acceptance.AcceptanceError,
                                        "canonicalization_pending=true"):
                acceptance.validate_raw_fixture_result(not_pending, parsed_state, contract)
            filled = json.loads(json.dumps(raw))
            filled["boundary_records"][0]["causal_prestate"] = {
                "work_ledger": {}, "resources": {}, "directory": {},
                "capacity": {}, "obligations": {}, "history": {}}
            with self.assertRaisesRegex(acceptance.AcceptanceError,
                                        "leave causal_prestate/planner_state null"):
                acceptance.validate_raw_fixture_result(filled, parsed_state, contract)
            missing_frames = json.loads(json.dumps(raw))
            missing_frames["audit_records"] = missing_frames["audit_records"][:1]
            with self.assertRaisesRegex(acceptance.AcceptanceError,
                                        "pair 1:1 in commit order"):
                acceptance.validate_raw_fixture_result(missing_frames, parsed_state,
                                                       contract)
            bad_producer = json.loads(json.dumps(raw))
            bad_producer["audit_records"][0]["frame"]["producer"] = "python"
            with self.assertRaisesRegex(acceptance.AcceptanceError,
                                        "producer must be cpp_online"):
                acceptance.validate_raw_fixture_result(bad_producer, parsed_state,
                                                       contract)

    def test_canonicalize_fixture_result_requires_acceptance_module(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state, contract, result = self._bundle(root)
            state_path = root / "state.json"
            state_path.write_text("{}", encoding="utf-8")
            parsed_state = acceptance.validate_fixture_state(state, state_path)
            source_root = root / "source-copy"
            package = (source_root / "sh_test_mesh/workload/llama2_7b_inference"
                       / "joint")
            package.mkdir(parents=True)
            (package / "__init__.py").write_text("", encoding="utf-8")
            try:
                with self.assertRaisesRegex(acceptance.AcceptanceError,
                                            "acceptance planner is missing"):
                    acceptance.canonicalize_fixture_result(result, parsed_state,
                                                           source_root)
                stub = (package / "acceptance_projection.py")
                stub.write_text(
                    "def canonicalize_raw_frames(raw_result, fixture_state,"
                    " *, source_root):\n"
                    "    filled = dict(raw_result)\n"
                    "    filled['canonicalization_pending'] = False\n"
                    "    return filled\n", encoding="utf-8")
                sys.modules.pop("joint", None)
                sys.modules.pop("joint.acceptance_projection", None)
                canonical = acceptance.canonicalize_fixture_result(result, parsed_state,
                                                                   source_root)
                self.assertFalse(canonical["canonicalization_pending"])
                sys.modules.pop("joint", None)
                sys.modules.pop("joint.acceptance_projection", None)
                partial = (package / "acceptance_projection.py")
                partial.write_text(
                    "def canonicalize_raw_frames(raw_result, fixture_state,"
                    " *, source_root):\n"
                    "    filled = dict(raw_result)\n"
                    "    filled['canonicalization_pending'] = True\n"
                    "    return filled\n", encoding="utf-8")
                with self.assertRaisesRegex(acceptance.AcceptanceError,
                                            "still pending canonicalization"):
                    acceptance.canonicalize_fixture_result(result, parsed_state,
                                                           source_root)
            finally:
                sys.modules.pop("joint", None)
                sys.modules.pop("joint.acceptance_projection", None)
                sys.path[:] = [item for item in sys.path
                               if not item.startswith(str(root))]

    def test_canonicalize_allows_same_root_reload_rejects_foreign_root(self):
        """D1 回归：joint 已加载时，同根二次 canonicalize 必须通过
        （旧 guard 把 joint.__path__（包目录）与父目录比较，永假，
        任何第二次调用都误报 import collision）；异根仍拒绝。"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state, contract, result = self._bundle(root)
            state_path = root / "state.json"
            state_path.write_text("{}", encoding="utf-8")
            parsed_state = acceptance.validate_fixture_state(state, state_path)
            module_text = (
                "def canonicalize_raw_frames(raw_result, fixture_state,"
                " *, source_root):\n"
                "    filled = dict(raw_result)\n"
                "    filled['canonicalization_pending'] = False\n"
                "    return filled\n")
            source_root = root / "source-copy"
            package = (source_root / "sh_test_mesh/workload/llama2_7b_inference"
                       / "joint")
            package.mkdir(parents=True)
            (package / "__init__.py").write_text("", encoding="utf-8")
            (package / "acceptance_projection.py").write_text(
                module_text, encoding="utf-8")
            other_root = root / "other-copy"
            other_package = (other_root / "sh_test_mesh/workload/"
                             "llama2_7b_inference/joint")
            other_package.mkdir(parents=True)
            (other_package / "__init__.py").write_text("", encoding="utf-8")
            (other_package / "acceptance_projection.py").write_text(
                module_text, encoding="utf-8")
            try:
                sys.modules.pop("joint", None)
                sys.modules.pop("joint.acceptance_projection", None)
                first = acceptance.canonicalize_fixture_result(result, parsed_state,
                                                               source_root)
                self.assertFalse(first["canonicalization_pending"])
                # joint 包此刻已加载：同根重载不得误报 collision。
                second = acceptance.canonicalize_fixture_result(result, parsed_state,
                                                                source_root)
                self.assertFalse(second["canonicalization_pending"])
                # 已加载异根 joint 时必须显式拒绝并要求 fresh process。
                with self.assertRaisesRegex(acceptance.AcceptanceError,
                                            "different joint package"):
                    acceptance.canonicalize_fixture_result(result, parsed_state,
                                                           other_root)
            finally:
                sys.modules.pop("joint", None)
                sys.modules.pop("joint.acceptance_projection", None)
                sys.path[:] = [item for item in sys.path
                               if not item.startswith(str(root))]

    def test_first_boundary_is_one_per_cluster_request_stage(self):
        rows = [
            {"cluster_id": "c", "request_id": "r", "stage": "P",
             "boundary_ordinal": 1, "marker": "later"},
            {"cluster_id": "c", "request_id": "r", "stage": "P",
             "boundary_ordinal": 0, "marker": "first"},
            {"cluster_id": "c", "request_id": "r", "stage": "D",
             "boundary_ordinal": 0, "marker": "decode"},
        ]
        selected = acceptance.first_boundary_samples(rows)
        self.assertEqual([row["marker"] for row in selected], ["decode", "first"])


class TraceAndContractTests(unittest.TestCase):
    def test_campaign_recipe_uses_token_horizon_and_balances_focus_by_context(self):
        for scenario in acceptance.PRICEABLE_SCENARIOS:
            by_context = {}
            for seed in range(64):
                row = acceptance._campaign_case_recipe(scenario, seed)
                causal = row["causal_input"]
                self.assertIn(causal["horizon_tokens"], (1, 8, 32, 128))
                self.assertNotIn("horizon_ns", causal)
                by_context.setdefault(causal["context_tokens"], set()).add(
                    causal["focus_stage"])
            self.assertTrue(all(stages == {"P", "D"}
                                for stages in by_context.values()),
                            f"scenario M{scenario:02d} lacks a stage in a context bucket")

    def test_fixture_state_rejects_nanosecond_horizon_alias(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state, _, _ = ColdFixtureContractTests()._bundle(root)
            state["causal_input"]["horizon_ns"] = state["causal_input"].pop(
                "horizon_tokens")
            asset = root / "system.json"
            asset.write_text("{}\n", encoding="utf-8")
            state["assets"][0]["sha256"] = acceptance.sha256_file(asset)
            with self.assertRaisesRegex(acceptance.AcceptanceError,
                                        "lacks fixed horizon_tokens"):
                acceptance.validate_fixture_state(state, root / "state.json")

    def test_first_second_selector_keeps_complete_selected_turns_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            trace = Path(tmp) / "trace.csv"
            trace.write_text(
                "arrival_time,session_id,request_id,prefix_len,prefill_length,decode_length,human_time,tool_time\n"
                "950000000,0,0,1,1,4,,50000000\n"
                ",0,1,1,1,8,,20000000\n"
                ",0,2,1,1,9,,\n"
                "1000000000,1,0,1,1,3,,\n", encoding="utf-8")
            selection = acceptance.select_first_second_trace(trace)
            self.assertEqual([(row["session_id"], row["turn"], row["arrival_ns"])
                              for row in selection["selected_requests"]],
                             [("session_0", 0, 950000000)])
            self.assertEqual(selection["first_excluded_turns"][0]["derived_arrival_ns"],
                             1000000000)
            self.assertEqual(selection["first_later_session"]["session_id"], "session_1")

    def test_first_second_selector_preserves_dictreader_column_mapping(self):
        with tempfile.TemporaryDirectory() as tmp:
            trace = Path(tmp) / "trace.csv"
            trace.write_text(
                "arrival_time,session_id,session_id,request_id,prefix_len,"
                "prefill_length,decode_length,human_time,tool_time\n"
                "950000000,ignored,0,0,1,1,4,,,extra-cell\n"
                ",ignored,0,1,1,1,8\n"
                "1000000000,ignored,later,0,1,1,3,,\n",
                encoding="utf-8",
            )

            selection = acceptance.select_first_second_trace(trace)

            self.assertEqual(
                [(row["session_id"], row["turn"], row["source_request_id"],
                  row["arrival_ns"], row["source_line"], row["decode_length"])
                 for row in selection["selected_requests"]],
                [("session_0", 0, "0", 950000000, 2, 4),
                 ("session_0", 1, "1", 950000000, 3, 8)],
            )
            self.assertEqual(selection["first_later_session"]["session_id"],
                             "session_later")

    def test_first_second_selector_validates_shuffled_tail_without_selecting_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            trace = Path(tmp) / "trace.csv"
            header = (
                "arrival_time,session_id,request_id,prefix_len,prefill_length,"
                "decode_length,human_time,tool_time"
            )
            rows = [
                "950000000,0,0,1,1,4,,50000000",
                "1000000000,later-z,,1,1,not-a-number,,",
            ]
            # Unique IDs deliberately arrive in a non-sorted order. Once the
            # first outside-window session is found, later row payloads are
            # irrelevant to selection, while their session IDs remain subject
            # to the whole-file contiguity check.
            tail_ids = [f"tail-{index:02d}" for index in range(32)]
            tail_ids = tail_ids[::2] + tail_ids[1::2][::-1]
            rows.extend(
                f"not-a-number,{session_id},,1,1,also-not-a-number,,"
                for session_id in tail_ids
            )
            trace.write_text("\n".join([header, *rows]) + "\n",
                             encoding="utf-8")

            selection = acceptance.select_first_second_trace(trace)

            self.assertEqual(selection["selected_requests"], [{
                "session_id": "session_0", "turn": 0,
                "request_id": "session_0_request_0",
                "source_request_id": "0", "arrival_ns": 950000000,
                "source_line": 2, "decode_length": 4,
            }])
            self.assertEqual(selection["first_later_session"], {
                "session_id": "session_later-z", "source_line": 3,
                "arrival_ns": 1000000000,
                "reason": "session first arrival >= window_ns",
            })
            self.assertEqual(selection["last_included_session"], "session_0")
            self.assertEqual(selection["source_sha256"], _sha(trace))

    def test_first_second_selector_rejects_noncontiguous_tail_after_cutoff(self):
        with tempfile.TemporaryDirectory() as tmp:
            trace = Path(tmp) / "trace.csv"
            trace.write_text(
                "arrival_time,session_id,request_id,prefix_len,prefill_length,"
                "decode_length,human_time,tool_time\n"
                # Selection-only validation happens after the old full-file
                # SID pass, so this must not mask a later structural failure.
                "950000000,0,,1,1,4,,\n"
                "1000000000,later,0,1,1,4,,\n"
                "not-a-number,other,0,1,1,not-a-number,,\n"
                "not-a-number,later,0,1,1,not-a-number,,\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                    acceptance.AcceptanceError,
                    "trace session is not contiguous: session_later"):
                acceptance.select_first_second_trace(trace)

    def test_frozen_trace_selection_reproduces_eight_requests(self):
        evidence = Path("/home/sunhao/joint_accel_execution_20261005.RCuiy5")
        trace = evidence / "manifests/astra_compute_20.csv"
        if not trace.is_file():
            self.skipTest("frozen evidence package is not mounted")
        frozen = acceptance.read_json(
            evidence / "inputs/trace-window-1s-selection.json")
        selection = acceptance.select_first_second_trace(trace)
        self.assertEqual(selection["selected_request_count"], 8)
        self.assertEqual(selection["selected_requests"][-1]["source_line"], 70)
        self.assertEqual(selection["first_excluded_turns"][-1]["derived_arrival_ns"],
                         1_045_441_000)
        self.assertEqual(selection["first_later_session"]["source_line"], 218)
        self.assertEqual(frozen["source_sha256"],
                         "56f2ddd8e845c69bf1361bd68fa0b68003cf73c7df77313c3b379de6d074cd97")
        for key in (
                "source_csv", "source_sha256", "window_ns", "selection",
                "selected_session_count", "selected_request_count",
                "selected_requests", "first_excluded_turns",
                "first_later_session", "last_included_session"):
            self.assertEqual(selection[key], frozen[key], key)

    def test_contract_unknown_key_and_k1_noncanonical_grid_are_rejected(self):
        contract = {
            "projection_abi_version": 1, "descriptor_version": 1,
            "pipeline_protocol_version": 1,
            "model_version": "v7-three-actions-20261014",
            "timer_whitelist_version": "timer-whitelist-v1-20261005",
            "window_tile_limit": 1, "window_local_error_ppm": 10000,
            "window_refine_margin_ppm": 20000,
            "prediction_config_sha256": "b" * 64,
        }
        self.assertEqual(acceptance.validate_kernel_contract(contract), contract)
        extra = dict(contract, unexpected=True)
        with self.assertRaisesRegex(acceptance.AcceptanceError, "keys mismatch"):
            acceptance.validate_kernel_contract(extra)
        invalid = dict(contract, window_local_error_ppm=5000)
        with self.assertRaisesRegex(acceptance.AcceptanceError, "unique 10000/20000"):
            acceptance.validate_kernel_contract(invalid)

    def test_resolved_v3_config_derives_the_exact_default_kernel_contract(self):
        # 发布默认 = F 链验收档（2026-10-09，PredictionParams 类默认翻转
        # 为 K2/ppm0/margin20000；exact 口径经 k=1 显式覆盖派生）。
        contract, record = acceptance.resolve_prediction_contract(
            acceptance.REPO_ROOT, None)
        self.assertEqual((contract["window_tile_limit"],
                          contract["window_local_error_ppm"],
                          contract["window_refine_margin_ppm"]), (2, 0, 20000))
        exact_contract, _ = acceptance.resolve_prediction_contract(
            acceptance.REPO_ROOT, None, k=1, local_error_ppm=10000,
            refine_margin_ppm=20000)
        self.assertEqual((exact_contract["window_tile_limit"],
                          exact_contract["window_local_error_ppm"],
                          exact_contract["window_refine_margin_ppm"]), (1, 10000, 20000))
        self.assertEqual(contract["prediction_config_sha256"], record["semantic_sha256"])
        self.assertEqual(len(contract), 9)


class AccuracyAndPerformanceTests(unittest.TestCase):
    def test_frozen_statistical_edge_reference_all_cases_pass(self):
        reference = Path("/home/sunhao/joint_accel_execution_20261005.RCuiy5/manifests"
                        "/acceptance-statistical-reference-v1.json")
        if not reference.is_file():
            self.skipTest("frozen statistical reference is not mounted")
        self.assertTrue(acceptance.statistical_reference_result(
            acceptance.read_json(reference))["pass"])

    def test_holdout_coverage_counts_independent_cluster_requests(self):
        rows = []
        for stage, cluster, context, competition, sources, capacity in (
                ("P", "c0", 128, "low", 1, "normal"),
                ("D", "c1", 2048, "high", 2, "stress")):
            rows.append({
                "cluster_id": cluster, "request_id": "r", "stage": stage,
                "boundary_ordinal": 0,
                "causal_metadata": {"context_tokens": context,
                                    "competition_level": competition,
                                    "source_count": sources},
                "capacity": {"pressure_bucket": capacity},
            })
        report = acceptance.sample_coverage(rows, min_stage_snapshots=1,
                                            min_bucket_snapshots=1)
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["stage_coverage"]["P"]["independent_clusters"], 1)
        self.assertEqual(report["major_bucket_coverage"]["capacity"],
                         {"normal": 1, "stress": 1})

    def test_l2_pairs_service_and_settle_separately_and_protect_charges(self):
        def result(service: int, settle: int, charge: int):
            return {"case_id": "M01-seed001", "cluster_id": "M01/seed001",
                "measurements": [{
                "stage": "D", "boundary_ordinal": 0,
                "fixed_instance_id": "decode-1", "fixed_action": "copy",
                "start_tick_ns": 100, "service_tick_ns": 100 + service,
                "settle_tick_ns": 100 + settle, "logical_event_count": 3,
                "charged_totals": {"bytes": charge, "ops": 1},
            }], "request_id": "r0", "initial_certificate": {"empty": True},
                "boundary_records": [{
                    "request_id": "r0", "stage": "D", "boundary_ordinal": 0,
                    "tick_ns": 100, "fixed_instance_id": "decode-1",
                    "fixed_action": "copy",
                    "causal_prestate": {"work_ledger": {"cursor": 1},
                                        "resources": {"link": {"remaining": 10}},
                                        "directory": {}, "capacity": {},
                                        "obligations": {}, "history": {}},
                    "causal_metadata": {"horizon_tokens": 100},
                    "planner_state": {"forecast_generator": {"identity": "g"}},
                }]}
        report = acceptance.compare_fixed_action_results(
            result(1000, 2000, 99), result(1010, 2020, 99))
        self.assertEqual(report["status"], "PASS")
        self.assertAlmostEqual(report["by_stage"]["D"]["service"]["mean"], 0.01)
        with self.assertRaisesRegex(acceptance.AcceptanceError, "charged totals"):
            acceptance.compare_fixed_action_results(
                result(1000, 2000, 99), result(1000, 2000, 100))

    def test_l2_post_prefix_boundary_state_mismatch_is_diagnostic_only(self):
        exact = {"case_id": "M01-seed001", "cluster_id": "M01/seed001",
                 "request_id": "r", "initial_certificate": {"cold": True},
                 "measurements": [{"stage": "P", "boundary_ordinal": 0,
                                   "fixed_instance_id": "i", "fixed_action": "copy",
                                   "start_tick_ns": 0, "service_tick_ns": 10,
                                   "settle_tick_ns": 11, "logical_event_count": 2,
                                   "charged_totals": {"bytes": 5}},
                                  {"stage": "D", "boundary_ordinal": 0,
                                   "fixed_instance_id": "i", "fixed_action": "copy",
                                   "start_tick_ns": 11, "service_tick_ns": 20,
                                   "settle_tick_ns": 21, "logical_event_count": 2,
                                   "charged_totals": {"bytes": 7}}],
                 "boundary_records": []}
        approx = json.loads(json.dumps(exact))
        for stage, tick, work in (("P", 0, 0), ("D", 12, 1)):
            exact["boundary_records"].append({
                "request_id": "r", "stage": stage, "boundary_ordinal": 0,
                "tick_ns": tick, "fixed_instance_id": "i", "fixed_action": "copy",
                "causal_prestate": {"work_ledger": {"cursor": work}, "resources": {},
                                    "directory": {}, "capacity": {},
                                    "obligations": {}, "history": {}},
                "causal_metadata": {"horizon_tokens": 32},
                "planner_state": {"forecast_generator": {"identity": "g"}},
            })
            approx["boundary_records"].append(json.loads(json.dumps(
                exact["boundary_records"][-1])))
        approx["boundary_records"][1]["causal_prestate"]["resources"] = {
            "link": {"remaining": 3}}
        report = acceptance.compare_fixed_action_results(exact, approx)
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["evaluable_pair_count"], 1)
        self.assertEqual(report["state_mismatches"][0]["stage"], "D")
        self.assertEqual(report["state_mismatches"][0]["reason"],
                         "canonical_causal_prestate_differs")

    @staticmethod
    def _projection_evaluation(exact_costs=(20, 40), approx_costs=(21, 19),
                               raw_winner="copy-b", final_winner="copy-a"):
        candidates = [
            {"candidate_id": "copy-a", "instance_id": "i-a", "action": "copy",
             "work_item": {"work_id": "w-a", "legs": [], "ready_at_ns": 0,
                           "depends_on_work": [], "is_candidate": True,
                           "service_leg_ids": None}},
            {"candidate_id": "copy-b", "instance_id": "i-b", "action": "recompute",
             "work_item": {"work_id": "w-b", "legs": [], "ready_at_ns": 0,
                           "depends_on_work": [], "is_candidate": True,
                           "service_leg_ids": None}},
        ]
        quality = {"approximation_profile": "exact", "window_count": 0,
                   "refinement_count": 0, "local_prefix_bound_max": 0,
                   "logical_event_count": 2, "physical_step_count": 2,
                   "bound_kind": "exact", "fully_exact": True,
                   "unsupported_reason": None}
        def rows(costs):
            return [{"candidate_id": item["candidate_id"], "status": "ok",
                     "service_ns": cost, "settle_ns": cost,
                     "quality": dict(quality),
                     "result": {
                         "status": "ok", "predicted_service_done_ns": 17 + cost,
                         "predicted_settle_done_ns": 17 + cost, "reason": None,
                         "dependency_attribution": [], "focus_bytes": 0,
                         "bytes_served_total": 0, "event_count": 2,
                         "resource_count": 0, "work_count": 1,
                         "closure_resource_count": 0, "closure_work_count": 1,
                         "approximations": [], "fallback": None,
                         "background_comparisons": [], **dict(quality),
                     }}
                    for item, cost in zip(candidates, costs)]
        generator_inputs = {"context_tokens": 128, "horizon_tokens": 128,
                            "generator_inputs": {}}
        generator = {"identity": "known-demand-v1", "causal_inputs": generator_inputs,
                     "semantic_sha256": acceptance._canonical_digest(
                         {"identity": "known-demand-v1", "causal_inputs": generator_inputs})}
        return {
            "version": "projection-evaluation-v1-20261006", "status": "PASS",
            "reason": None, "request_id": "r0", "stage": "P",
            "boundary_ordinal": 0, "tick_ns": 17, "horizon_tokens": 128,
            "projection_input": {"now_ns": 17, "resources": [{"key": "link:0"}]},
            "candidates": candidates, "inapplicable_candidates": [],
            "forecast_generator": generator,
            "approx_results": rows(approx_costs), "exact_results": rows(exact_costs),
            "arbitration": {"trigger_reason": "candidate_sensitive_band",
                            "exact_recomputed_ids": [final_winner],
                            "approximate_selected_id": raw_winner,
                            "exact_selected_id": "copy-a", "selected_id": final_winner,
                            "old_tie_order": ["copy-a", "copy-b"]},
            "library_sha256": "a" * 64, "prediction_config_sha256": "b" * 64,
            # D12：生产 DTO 的信息性键（比较器必需接受）。
            "model_semantics_version": "v7-three-actions-20261014",
            "world_source_status": "replayed",
        }

    def test_l3_scores_exact_and_approx_vectors_from_acceptance_projection_dto(self):
        boundary = {"request_id": "r0", "stage": "P", "boundary_ordinal": 0,
                    "tick_ns": 17, "causal_metadata": {"context_tokens": 128,
                                                          "horizon_tokens": 128,
                                                          "generator_inputs": {}}}
        dto = self._projection_evaluation()
        scored = acceptance.compare_projection_evaluation(
            dto, boundary, prediction_config_sha256="b" * 64,
            library_sha256="a" * 64)
        self.assertTrue(scored["evaluable"])
        self.assertEqual(scored["regret"]["approximate_selected_id"], "copy-b")
        self.assertAlmostEqual(scored["regret"]["regret"], 1.0)
        self.assertEqual(scored["policy_diagnostic"]["selected_id"], "copy-a")
        self.assertEqual(scored["policy_diagnostic"]["regret"], 0.0)
        self.assertEqual(scored["exact_recomputed_ids"], ["copy-a"])
        self.assertEqual(scored["inapplicable_candidates"], [])
        different_tick = dict(dto, tick_ns=18)
        with self.assertRaisesRegex(acceptance.AcceptanceError, "boundary mismatch"):
            acceptance.compare_projection_evaluation(
                different_tick, boundary, prediction_config_sha256="b" * 64)

    def test_l3_inapplicable_menu_rows_are_not_required_to_have_native_prices(self):
        boundary = {"request_id": "r0", "stage": "P", "boundary_ordinal": 0,
                    "tick_ns": 17, "causal_metadata": {"context_tokens": 128,
                                                          "horizon_tokens": 128,
                                                          "generator_inputs": {}}}
        dto = self._projection_evaluation()
        dto["inapplicable_candidates"] = [
            {"candidate_id": "stay-unavailable", "instance_id": "i-c",
             "action": "stay", "reason": "instance is not resident"},
            {"candidate_id": "recompute-unavailable", "instance_id": "i-d",
             "action": "recompute", "reason": "required weights are not available"},
        ]
        scored = acceptance.compare_projection_evaluation(
            dto, boundary, prediction_config_sha256="b" * 64,
            library_sha256="a" * 64)
        self.assertEqual(scored["candidate_order"], ["copy-a", "copy-b"])
        self.assertEqual(scored["inapplicable_candidates"], dto["inapplicable_candidates"])
        self.assertEqual(len(dto["approx_results"]), 2)
        self.assertEqual(len(dto["exact_results"]), 2)

    def test_l3_rejects_applicable_inapplicable_candidate_overlap(self):
        boundary = {"request_id": "r0", "stage": "P", "boundary_ordinal": 0,
                    "tick_ns": 17, "causal_metadata": {"context_tokens": 128,
                                                          "horizon_tokens": 128,
                                                          "generator_inputs": {}}}
        dto = self._projection_evaluation()
        dto["inapplicable_candidates"] = [
            {"candidate_id": "copy-a", "instance_id": "i-a", "action": "copy",
             "reason": "duplicate menu identity"},
        ]
        with self.assertRaisesRegex(acceptance.AcceptanceError,
                                    "unique and disjoint"):
            acceptance.compare_projection_evaluation(
                dto, boundary, prediction_config_sha256="b" * 64,
                library_sha256="a" * 64)

    def test_l3_rejects_untyped_inapplicable_action(self):
        boundary = {"request_id": "r0", "stage": "P", "boundary_ordinal": 0,
                    "tick_ns": 17, "causal_metadata": {"context_tokens": 128,
                                                          "horizon_tokens": 128,
                                                          "generator_inputs": {}}}
        dto = self._projection_evaluation()
        dto["inapplicable_candidates"] = [
            {"candidate_id": "invalid", "instance_id": "i-c", "action": [],
             "reason": "malformed action"},
        ]
        with self.assertRaisesRegex(acceptance.AcceptanceError, "invalid action"):
            acceptance.compare_projection_evaluation(
                dto, boundary, prediction_config_sha256="b" * 64,
                library_sha256="a" * 64)

    def test_l3_zero_best_positive_pre_refinement_choice_is_hard_failure(self):
        boundary = {"request_id": "r0", "stage": "P", "boundary_ordinal": 0,
                    "tick_ns": 17, "causal_metadata": {"context_tokens": 128,
                                                          "horizon_tokens": 128,
                                                          "generator_inputs": {}}}
        dto = self._projection_evaluation(exact_costs=(0, 2), approx_costs=(1, 0),
                                          raw_winner="copy-b", final_winner="copy-a")
        with self.assertRaisesRegex(acceptance.AcceptanceError,
                                    "positive cost over zero exact optimum"):
            acceptance.compare_projection_evaluation(
                dto, boundary, prediction_config_sha256="b" * 64)

    def test_l3_incomplete_evaluation_is_not_a_measurement(self):
        boundary = {"request_id": "r0", "stage": "D", "boundary_ordinal": 2,
                    "tick_ns": 99}
        dto = {"version": "projection-evaluation-v1-20261006", "status": "INCOMPLETE",
               "reason": "boundary lacks settled resource vector", "request_id": "r0",
               "stage": "D", "boundary_ordinal": 2, "tick_ns": 99}
        result = acceptance.compare_projection_evaluation(
            dto, boundary, prediction_config_sha256="b" * 64)
        self.assertFalse(result["evaluable"])
        self.assertEqual(result["status"], "INCOMPLETE")

    def test_performance_requires_complete_rotated_same_work_runs(self):
        orders = [
            ["old_exact", "new_exact", "new_approx"],
            ["new_exact", "new_approx", "old_exact"],
            ["new_approx", "old_exact", "new_exact"],
            ["old_exact", "new_exact", "new_approx"],
            ["new_exact", "new_approx", "old_exact"],
        ]
        runs = [{
            "status": "complete", "order": order,
            "arms": {name: {"status": "completed", "natural_completion": True,
                            "elapsed_host_ns": duration,
                            "work_signature_sha256": "c" * 64,
                            "peak_rss_bytes": 10}
                     for name, duration in (("old_exact", 400),
                                            ("new_exact", 100),
                                            ("new_approx", 50))},
        } for order in orders]
        summary = acceptance.summarize_performance_repetitions(runs)
        self.assertEqual(summary["speed_ratios"], {
            "S_engineering": 4.0, "S_approx_extra": 2.0, "S_total": 8.0})
        self.assertFalse(summary["speed_goal_met"])
        runs[-1]["arms"]["new_approx"]["natural_completion"] = False
        with self.assertRaisesRegex(acceptance.AcceptanceError,
                                    "partial/timeout performance work"):
            acceptance.summarize_performance_repetitions(runs)

    def test_exact_only_summary_never_claims_approximation_speed(self):
        orders = [["old_exact", "new_exact"], ["new_exact", "old_exact"],
                  ["old_exact", "new_exact"], ["new_exact", "old_exact"],
                  ["old_exact", "new_exact"]]
        runs = [{
            "status": "complete", "order": order,
            "arms": {name: {"status": "completed", "natural_completion": True,
                            "elapsed_host_ns": duration,
                            "work_signature_sha256": "d" * 64}
                     for name, duration in (("old_exact", 400), ("new_exact", 100))},
        } for order in orders]
        summary = acceptance.summarize_exact_only_repetitions(runs)
        self.assertEqual(summary["speed_ratios"], {
            "S_engineering": 4.0, "S_approx_extra": None, "S_total": None})
        self.assertIsNone(summary["speed_goal_met"])
        self.assertFalse(summary["formal_speed_claim"])
        runs[-1]["arms"]["new_exact"]["natural_completion"] = False
        with self.assertRaisesRegex(acceptance.AcceptanceError,
                                    "incomplete exact-only arm"):
            acceptance.summarize_exact_only_repetitions(runs)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
