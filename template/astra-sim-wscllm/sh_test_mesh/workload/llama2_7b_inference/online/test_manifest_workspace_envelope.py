"""Actual WSCLLM loader/materializer/emitter and C++ Plan capacity coverage.

蓝本 astra-sim-face-LRU online/test_manifest_workspace_envelope.py,逐条
适配 wscllm 命名/fixture(W 无 cost_oracle,推导/查询走仓内 build 发现序;
P 侧整段 prefill 经 _emit_prefill_stage 实发射,D 侧列车经 _plan_train/
emit_iteration_train/emit_train_first_step/emit_train_remainder 实发射)。
"""
from __future__ import annotations
import contextlib
import copy
import csv
import ctypes
from dataclasses import replace
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import generate_wsc_llm_trace as trace
import plan_materializer
from online.coarse_workspace_envelope import (
    DieWorkspaceEnvelopeClient, WorkspaceEnvelopeError, _read_owned_json,
    coarse_cabi_candidates,
)
from online.die_model_adapter import DieModelContext
from online.graph_batch_builder import GraphBatchBuilder, _prefix_of
from online.manifest_workspace_envelope import derive_manifest_workspace_envelope
from generate_wsc_llm_trace import _emit_prefill_stage
from online.wsc_llm_online_scheduler import (
    WscLlmOnlineScheduler, _OnlineRequestRuntime,
)


def _resolve_library() -> Path:
    env_value = os.environ.get("DIE_COARSE_CABI_PATH")
    candidates = [Path(env_value)] if env_value else []
    candidates.extend(coarse_cabi_candidates())
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        "libcoarse_die_cabi.so unavailable; build the coarse_die_cabi target "
        "or set DIE_COARSE_CABI_PATH; candidates: "
        + " -> ".join(str(path) for path in candidates))


class ManifestWorkspaceEnvelopeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="wscllm-capacity-certificate-")
        cls.addClassCleanup(cls.temporary.cleanup)
        root = Path(cls.temporary.name)
        sh_test = Path(__file__).resolve().parents[3]
        cls.library = _resolve_library()
        for directory in ("hardware", "system"):
            shutil.copytree(sh_test / directory, root / directory)
        queue = root / "requests.csv"
        with queue.open("w", newline="") as target:
            writer = csv.writer(target)
            writer.writerow(("session_id", "turn_index", "request_id", "prefill_length", "decode_length",
                             "session_arrival_time_ns", "inter_request_interval_ns", "description"))
            writer.writerows((("A", 0, "A0", 7, 3, 0, "", "fresh"),
                              ("A", 1, "A1", 9, 5, "", 1000, "history"),
                              ("B", 0, "B0", 6, 4, 1000, "", "mixed")))
        config_path = root / "trace_config.csv"
        with (sh_test / "workload/llama2_7b_inference/trace_config.csv").open(newline="", encoding="utf-8-sig") as source:
            reader = csv.DictReader(source)
            fieldnames, rows = reader.fieldnames, list(reader)
        updates = {"layers": "1", "hidden_size": "384", "ffn_size": "768", "num_heads": "12",
            "vocab_size": "384", "bytes_per_elem": "2", "prefill_chunk_size": "4",
            "request_queue_csv": str(queue), "request_queue_session_limit": "0",
            "hardware_config": str(root / "hardware/face_case5_config_c.json"),
            "system_template": str(root / "system/llama2_7b_die_template.json"),
            "trace_granularity": "request_aggregated", "remote_operand_loads": "false",
            "kv_reserve_context_tokens": "64", "position_encoding": "rope"}
        for row in rows:
            if row.get("kind") == "config" and row.get("key") in updates:
                row["value"] = updates[row["key"]]
        with config_path.open("w", newline="") as target:
            writer = csv.DictWriter(target, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        old_root = trace.SH_TEST_DIR
        try:
            trace.SH_TEST_DIR = root
            with patch.dict(os.environ, {"DIE_COARSE_CABI_PATH": str(cls.library)}):
                cls.config = trace.load_wsc_llm_trace_config(config_path)
        finally:
            trace.SH_TEST_DIR = old_root
        cls.manifest = dict(plan_materializer._resolved_backend_keys(cls.config))
        cls.manifest["requests"] = plan_materializer._derive_manifest_requests(cls.config, {})
        cls.manifest["manifest_source"] = "real-loader-derived-test-input"
        cls.context = DieModelContext.from_service(cls.manifest, cls.config)
        with patch.dict(os.environ, {"DIE_COARSE_CABI_PATH": str(cls.library)}):
            cls.certificates = {cap: derive_manifest_workspace_envelope(
                cls.manifest, cls.config, train_max_iter=cap, library_path=cls.library) for cap in (2, 0)}

    def test_proof_identity_frozen_bounds_and_constructor_cap_contract(self):
        certificate = self.certificates[2]
        proof = json.loads(certificate.canonical_json)
        self.assertEqual(proof["schema"], "wscllm_manifest_workspace_certificate_v1")
        self.assertEqual(proof["coverage_revision"],
                         "wscllm_train_one_prefill_all_request_decode_superset_v1")
        self.assertEqual(proof["capacity_model_revision"], "workspace_lifetimes_v2")
        self.assertEqual(len(proof["proof_tasks"]), 6 * 2 * 14)
        for record in proof["proof_tasks"]:
            self.assertIn("coarse_die_task_v2", record["request_json"])
            self.assertEqual(json.loads(record["response_json"])["capacity_model_revision"], "workspace_lifetimes_v2")
        with self.assertRaises(TypeError):
            certificate.bytes_by_rank[0] = 0
        graph = GraphBatchBuilder(self.config)
        scheduler = WscLlmOnlineScheduler(manifest=self.manifest, config=self.config, graph=graph,
            workspace_admission="required", train_max_iter=2,
            prospective_workspace_certificate=certificate.without_proof())
        snapshot = scheduler.kv_manager.workspace_capacity_snapshot()
        # W 与 wscllm-LRU manager 同族:容量快照保持 protocol v1 行,不带
        # prospective 注记;地板经 _fit_remaining/ensure_physical_fit 生效。
        self.assertEqual(snapshot["protocol_version"], 1)
        self.assertEqual(
            dict(scheduler.kv_manager.prospective_workspace_bytes_by_rank),
            dict(certificate.bytes_by_rank))
        self.assertTrue(all(
            "prospective_workspace_bytes" not in row for row in snapshot["ranks"]))
        self.assertEqual(scheduler._train_max_iter, 2)
        self.assertEqual(scheduler.workspace_envelope_certificate.canonical_json, "")
        self.assertEqual(scheduler.workspace_floor_reconciliation, None)
        with self.assertRaisesRegex(ValueError, "cap differs"):
            WscLlmOnlineScheduler(manifest=self.manifest, config=self.config, graph=graph,
                workspace_admission="required", train_max_iter=0,
                prospective_workspace_certificate=certificate)
        with self.assertRaisesRegex(ValueError, "requires workspace admission"):
            WscLlmOnlineScheduler(manifest=self.manifest, config=self.config, graph=graph,
                workspace_admission="off", train_max_iter=2,
                prospective_workspace_certificate=certificate.without_proof())

    def test_stale_certificate_rejects_different_manifest_model_and_hardware(self):
        from online.manifest_workspace_envelope import validate_manifest_workspace_certificate
        certificate = self.certificates[2].without_proof()
        with self.assertRaisesRegex(ValueError, "train cap differs"):
            validate_manifest_workspace_certificate(certificate, self.manifest, self.config, 0, self.context)
        manifest = copy.deepcopy(self.manifest)
        manifest["requests"][0]["request_id"] += "-changed"
        with self.assertRaisesRegex(ValueError, "manifest differs"):
            validate_manifest_workspace_certificate(certificate, manifest, self.config, 2, self.context)
        with self.assertRaisesRegex(ValueError, "configuration differs"):
            validate_manifest_workspace_certificate(certificate, self.manifest,
                replace(self.config, hidden_size=self.config.hidden_size * 2), 2, self.context)
        context = DieModelContext.from_service(self.manifest, self.config)
        context.config_hash = "0" * 64
        with self.assertRaisesRegex(ValueError, "resolved hardware differs"):
            validate_manifest_workspace_certificate(certificate, self.manifest, self.config, 2, context)

    def test_bad_manifest_and_producer_inputs_fail_closed(self):
        empty = copy.deepcopy(self.manifest)
        empty["requests"] = []
        cases = [empty]
        for key, value in (("prefill_length", True), ("decode_length", 1.0),
                           ("final_context_tokens", 1 << 64), ("queue_index", -1),
                           ("history_tokens_before", 1), ("turn_index", 2)):
            invalid = copy.deepcopy(self.manifest)
            invalid["requests"][0][key] = value
            cases.append(invalid)
        duplicate = copy.deepcopy(self.manifest)
        duplicate["requests"][1]["request_id"] = duplicate["requests"][0]["request_id"]
        cases.append(duplicate)
        for invalid in cases:
            with self.assertRaises(WorkspaceEnvelopeError):
                derive_manifest_workspace_envelope(invalid, self.config, train_max_iter=2, library_path=self.library)
        for cap in (True, -1, 1.0, 1 << 64):
            with self.assertRaises(WorkspaceEnvelopeError):
                derive_manifest_workspace_envelope(self.manifest, self.config, train_max_iter=cap, library_path=self.library)
        with self.assertRaisesRegex(WorkspaceEnvelopeError, "partition"):
            derive_manifest_workspace_envelope(self.manifest,
                replace(self.config, inference_groups=(self.config.inference_groups[0],)),
                train_max_iter=2, library_path=self.library)
        with self.assertRaisesRegex(WorkspaceEnvelopeError, "complete loaded request queue"):
            derive_manifest_workspace_envelope(
                {"requests": self.manifest["requests"][:-1]}, self.config,
                train_max_iter=2, library_path=self.library)

    def test_actual_emitter_plans_fit_capped_uncapped_history_and_split_subsets(self):
        evidence = []
        certificate = self.certificates[2]
        uncapped = self.certificates[0]
        with DieWorkspaceEnvelopeClient(self.context.die_config_path, self.library) as client:
            lib = client._library
            lib.die_coarse_compile_batch_v2.argtypes = [ctypes.c_uint64, ctypes.c_char_p]
            lib.die_coarse_compile_batch_v2.restype = ctypes.c_void_p
            lib.die_coarse_release_plan_v2.argtypes = [ctypes.c_uint64]
            lib.die_coarse_release_plan_v2.restype = ctypes.c_int

            def compile_and_check(graph, label):
                nodes = [node for node in graph.batch["nodes"] if "die_op_spec" in node["compute"]]
                self.assertTrue(nodes)
                specs = [node["compute"]["die_op_spec"] for node in nodes]
                result = _read_owned_json(lib, lib.die_coarse_compile_batch_v2(
                    client.model_handle, json.dumps({"specs": specs}).encode()), "actual emitted compile")
                handles = [row["plan_handle"] for row in result["results"] if row is not None]
                try:
                    self.assertTrue(all(row is not None for row in result["results"]), result.get("errors"))
                    for node, compiled in zip(nodes, result["results"]):
                        # Selected immutable Plan summary, not another capacity bound.
                        peak = compiled["summary"]["hbm_workspace_peak_bytes"]
                        self.assertLessEqual(peak, certificate.bytes_by_rank[node["rank"]])
                        self.assertLessEqual(peak, uncapped.bytes_by_rank[node["rank"]])
                        evidence.append({"rank": node["rank"], "label": label,
                            "op": node["compute"]["die_op_spec"]["op_kind"],
                            "plan_hash": compiled["plan_hash"], "workspace_bytes": peak})
                finally:
                    for handle in handles:
                        self.assertEqual(lib.die_coarse_release_plan_v2(handle), 0)

            # Actual P-side producer: whole-segment chunked prefill per request
            # (fresh graph per request so earlier spans never leak into the
            # next request's compile), plus single-chunk subsets.
            p_group = next(
                group for group in self.config.inference_groups
                if group.phase_role == "prefill")
            for row in self.manifest["requests"]:
                spans = []
                processed = 0
                while processed < row["prefill_length"]:
                    chunk = min(self.config.prefill_chunk_size, row["prefill_length"] - processed)
                    spans.append((chunk, row["history_tokens_before"] + processed + chunk))
                    processed += chunk
                for part in (spans, spans[:1], spans[1:]):
                    if not part:
                        continue
                    graph = GraphBatchBuilder(self.config)
                    graph.set_die_context(DieModelContext.from_service(self.manifest, self.config))
                    graph.begin_batch()
                    _emit_prefill_stage(config=self.config, builders=graph.builders,
                        group=p_group, prefix=_prefix_of(row), stage="current_prefill",
                        tokens=row["prefill_length"], initial_context_tokens=row["history_tokens_before"],
                        die_ctx=graph.die_context)
                    graph._collect()
                    compile_and_check(graph, f"prefill:{row['request_id']}")

            # Actual D-side producer: complete cohort/offset/cap trains through
            # the scheduler's own plan + graph emission, split or whole.
            by_session = {}
            for row in self.manifest["requests"]:
                by_session.setdefault(row["session_id"], []).append(row)
            choices = [[None, *rows] for rows in by_session.values()]
            from itertools import product
            for first_token_split in ("0", "1"):
                with patch.dict(os.environ, {
                    "DIE_COARSE_CABI_PATH": str(self.library),
                    "SH_FIRST_TOKEN_SPLIT": first_token_split,
                    "SH_TRAIN_MAX_ITER": "8",
                }):
                    graph = GraphBatchBuilder(self.config)
                    scheduler = WscLlmOnlineScheduler(
                        manifest=self.manifest, config=self.config, graph=graph,
                        workspace_admission="off")
                    graph.set_die_context(scheduler.die_model_context)
                    state = scheduler.instances[next(
                        index for index, item in enumerate(self.config.inference_groups)
                        if item.phase_role == "decode")]
                    for selected in product(*choices):
                        cohort = [row for row in selected if row is not None]
                        if not cohort:
                            continue
                        for consumed in product(*(
                                range(row["decode_length"]) for row in cohort)):
                            runtimes = []
                            for row, offset in zip(cohort, consumed):
                                runtime = _OnlineRequestRuntime(dict(row))
                                runtime.decode_tokens_consumed = offset
                                runtime.decode_instance_index = state.index
                                runtimes.append(runtime)
                            state.active_decode = runtimes
                            state.active_decode_lookup = set(runtimes)
                            for cap in (0, 2):
                                scheduler._train_max_iter = cap
                                plan = scheduler._plan_train(state)
                                self.assertIsNotNone(plan)
                                graph.begin_batch()
                                graph_plan = {
                                    "train_id": plan["train_id"],
                                    "instance_index": state.index,
                                    "joiners": [],
                                    "pass_spans": list(plan["pass_spans"]),
                                    "iterations": int(plan["iterations"]),
                                    "exit_members": [
                                        {"request_id": runtime.request_id,
                                         "session_id": runtime.session_id}
                                        for runtime in plan["exit_members"]],
                                }
                                split = scheduler._first_token_plan(plan, [])
                                if split is not None and split["split"]:
                                    graph_plan["first_token"] = split
                                    graph.emit_train_first_step(graph_plan)
                                    first_specs = len([
                                        node for node in graph.batch["nodes"]
                                        if "die_op_spec" in node["compute"]])
                                    graph.emit_train_remainder(graph_plan)
                                    all_nodes = [
                                        node for node in graph.batch["nodes"]
                                        if "die_op_spec" in node["compute"]]
                                    self.assertTrue(all_nodes[first_specs:])
                                else:
                                    if split is not None:
                                        graph_plan["first_token"] = split
                                    graph.emit_iteration_train(graph_plan)
                                compile_and_check(graph, f"decode:cap={cap}")
        self.assertTrue(evidence)


if __name__ == "__main__":
    unittest.main()
