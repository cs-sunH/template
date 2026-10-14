"""Real-C-ABI checks for the finite LRU manifest workspace certificate."""

from __future__ import annotations

from collections import Counter
import csv
from dataclasses import replace
from itertools import product
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


_ONLINE_DIR = Path(__file__).resolve().parent
_WORKLOAD_DIR = _ONLINE_DIR.parent
_REPO_DIR = _ONLINE_DIR.parents[3]
_ROOT_DIR = _ONLINE_DIR.parents[4]
_PROTO_ROOT = _REPO_DIR / "work" / "chakra-python-root"
_REPRO_FIXTURE = (
    _ROOT_DIR / "die" / "docs" / "analysis" /
    "lru-workspace-v2-20261014" / "online-reproduction" /
    "work-layout" / "work" / "lru-online-workspace-20261014" /
    "e2e-large-workspace-probe"
)
_LIBRARY = Path(os.environ.get(
    "DIE_WORKSPACE_ENVELOPE_TEST_LIBRARY",
    os.environ.get("DIE_COARSE_CABI_PATH", str(
        _ROOT_DIR / "work" / "build-die" / "lib" / "libcoarse_die_cabi.so"))))

for _path in (_PROTO_ROOT, _REPO_DIR / "sh_test_mesh", _WORKLOAD_DIR,
              _ONLINE_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import generate_wsc_llm_trace as trace_module  # noqa: E402
from generate_wsc_llm_trace import _emit_prefill_stage  # noqa: E402
import plan_materializer  # noqa: E402
from online.die_model_adapter import DieModelContext  # noqa: E402
from online.die_workspace_envelope import (  # noqa: E402
    DieWorkspaceEnvelopeClient,
    WorkspaceEnvelopeError,
)
from online.graph_batch_builder import GraphBatchBuilder, _prefix_of  # noqa: E402
from online.manifest_workspace_envelope import (  # noqa: E402
    derive_manifest_workspace_envelope,
)
from online.online_service import _workspace_admission_mode  # noqa: E402
from online.wsc_llm_online_scheduler import (  # noqa: E402
    WscLlmOnlineScheduler,
    _OnlineRequestRuntime,
)


def _canonical_task_bytes(task: dict) -> bytes:
    return json.dumps(task, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def _task_nodes(graph: GraphBatchBuilder) -> list[tuple[int, dict]]:
    return [
        (int(node["rank"]), node["compute"]["die_op_spec"])
        for node in graph.batch["nodes"]
        if isinstance(node.get("compute"), dict)
        and isinstance(node["compute"].get("die_op_spec"), dict)
    ]


def _empty_delta(ranks: tuple[int, ...]) -> dict:
    return {
        "schema_version": 2,
        "delivery_sequence": 0,
        "delivery_epoch": 0,
        "tick": 0,
        "deferred_from_tick": 0,
        "reasons": ["TICK"],
        "arrivals": [],
        "completed_groups": [],
        "completed_nodes": [],
        "affected_ranks": [],
        "snapshot_handle": {"epoch": 0, "tick": 0},
        "workspace_state": {
            "protocol_version": 1,
            "ranks": [
                {"rank": rank, "workspace_revision": 0,
                 "active_workspace_bytes": 0}
                for rank in ranks
            ],
        },
    }


class ManifestWorkspaceEnvelopeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # A missing fixture is a checkout error, not a skip. Only the external
        # compiled C ABI may be unavailable in a Python-only run.
        if not _LIBRARY.is_file():
            raise unittest.SkipTest(
                "real workspace-capacity C ABI is unavailable; set "
                f"DIE_WORKSPACE_ENVELOPE_TEST_LIBRARY or DIE_COARSE_CABI_PATH: "
                f"{_LIBRARY}")
        seed_files = (
            _REPRO_FIXTURE / "hardware_coarse_8rank.json",
            _REPRO_FIXTURE / "system_coarse_template.json",
            _REPRO_FIXTURE / "trace_config.csv",
        )
        missing = [str(path) for path in seed_files if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "tracked online-reproduction fixture seed is missing: "
                + ", ".join(missing))

        cls._old_sh_test_dir = trace_module.SH_TEST_DIR
        cls._temporary = tempfile.TemporaryDirectory(
            prefix="lru-manifest-envelope-test.")
        cls._temp_root = Path(cls._temporary.name)
        cls.addClassCleanup(cls._cleanup_fixture)
        fixture_dir = cls._temp_root / "fixture"
        fixture_dir.mkdir(parents=True)
        hardware_path = fixture_dir / "hardware_coarse_8rank.json"
        system_path = fixture_dir / "system_coarse_template.json"
        trace_path = fixture_dir / "trace_config.csv"
        queue_path = fixture_dir / "manifest_queue.csv"
        hardware_path.write_bytes(seed_files[0].read_bytes())
        system_path.write_bytes(seed_files[1].read_bytes())

        # Retain the official four-request shapes while constructing the
        # queue as explicit test input (two legal, independent two-turn
        # sessions; arrival and inter-turn intervals are 1000 ns multiples).
        with queue_path.open("w", newline="", encoding="utf-8") as target:
            writer = csv.writer(target, lineterminator="\n")
            writer.writerow((
                "session_id", "turn_index", "request_id", "prefill_length",
                "decode_length", "session_arrival_time_ns",
                "inter_request_interval_ns", "description"))
            writer.writerows((
                ("session_manifest_a", 0, "manifest_a_turn0", 64, 2,
                 0, "", "seed turn"),
                ("session_manifest_a", 1, "manifest_a_turn1", 700, 5,
                 "", 1000, "history turn"),
                ("session_manifest_b", 0, "manifest_b_turn0", 96, 3,
                 1000, "", "seed turn"),
                ("session_manifest_b", 1, "manifest_b_turn1", 900, 3,
                 "", 2000, "history turn"),
            ))

        rewritten_paths = {
            "output_dir": str((fixture_dir / "unused_output").resolve()),
            "request_queue_csv": str(queue_path.resolve()),
            "hardware_config": str(hardware_path.resolve()),
            "system_template": str(system_path.resolve()),
        }
        with seed_files[2].open(newline="", encoding="utf-8") as source:
            reader = csv.DictReader(source)
            fieldnames = list(reader.fieldnames or ())
            trace_rows = list(reader)
        for row in trace_rows:
            if row.get("kind") == "config" and row.get("key") in rewritten_paths:
                row["value"] = rewritten_paths[row["key"]]
        with trace_path.open("w", newline="", encoding="utf-8") as target:
            writer = csv.DictWriter(target, fieldnames=fieldnames,
                                    lineterminator="\n")
            writer.writeheader()
            writer.writerows(trace_rows)

        # The production resolver writes generated runtime config under this
        # temporary root. Restore the module global after class teardown.
        trace_module.SH_TEST_DIR = (cls._temp_root / "trace_root").resolve()
        with patch.dict(os.environ, {"DIE_COARSE_CABI_PATH": str(_LIBRARY)}):
            cls.config = trace_module.load_wsc_llm_trace_config(trace_path)
            manifest_rows = plan_materializer._derive_manifest_requests(
                cls.config, {})
            resolved_system = json.loads(
                cls.config.system_config.read_text(encoding="utf-8"))
        cls.manifest = {
            "compute-cost-model": "die",
            "die-model-config": resolved_system["die-model-config"],
            "manifest_source": "test-derived-finite-four-request-fixture",
            "repo_variant": "astra-sim-wscllm-LRU",
            "requests": manifest_rows,
        }
        if len(cls.manifest["requests"]) != 4:
            raise AssertionError("loader/materializer did not derive four requests")
        cls.context = DieModelContext.from_service(cls.manifest, cls.config)
        if cls.context is None:
            raise AssertionError("official fixture did not select the DIE backend")
        cls.p_group = next(
            group for group in cls.config.inference_groups
            if group.phase_role == "prefill")
        cls.d_group = next(
            group for group in cls.config.inference_groups
            if group.phase_role == "decode")
        with patch.dict(os.environ, {"DIE_COARSE_CABI_PATH": str(_LIBRARY)}):
            cls.certificate = derive_manifest_workspace_envelope(
                cls.manifest, cls.config, library_path=_LIBRARY)

    @classmethod
    def _cleanup_fixture(cls) -> None:
        trace_module.SH_TEST_DIR = cls._old_sh_test_dir
        cls._temporary.cleanup()

    def _client(self):
        return DieWorkspaceEnvelopeClient(
            self.context.die_config_path, library_path=_LIBRARY)

    def _actual_prefill_specs(self) -> list[tuple[int, dict]]:
        actual = []
        for row in self.manifest["requests"]:
            graph = GraphBatchBuilder(self.config)
            graph.set_die_context(self.context)
            graph.begin_batch()
            mark = graph._mark()
            _emit_prefill_stage(
                config=self.config,
                builders=graph.builders,
                group=self.p_group,
                prefix=_prefix_of(row),
                stage="current_prefill",
                tokens=row["prefill_length"],
                initial_context_tokens=row["history_tokens_before"],
                die_ctx=self.context,
            )
            graph._collect(mark)
            actual.extend(_task_nodes(graph))
        return actual

    def _scheduler_for_train_checks(self):
        graph = GraphBatchBuilder(self.config)
        scheduler = WscLlmOnlineScheduler(
            manifest=self.manifest,
            config=self.config,
            graph=graph,
            workspace_admission="off",
        )
        graph.set_die_context(scheduler.die_model_context)
        return scheduler, graph, scheduler.instances[
            next(index for index, item in enumerate(self.config.inference_groups)
                 if item.phase_role == "decode")]

    @staticmethod
    def _graph_train_plan(plan: dict, state) -> dict:
        return {
            "train_id": plan["train_id"],
            "instance_index": state.index,
            "joiners": [],
            "pass_spans": list(plan["pass_spans"]),
            "iterations": int(plan["iterations"]),
            "sentinel": bool(plan["sentinel"]),
            "exit_members": [
                {"request_id": runtime.request_id,
                 "session_id": runtime.session_id}
                for runtime in plan["exit_members"]
            ],
        }

    def _assert_task_covered(self, client, rank: int, spec: dict,
                             d_bound_by_rank_op: dict,
                             queried: dict[bytes, object]) -> None:
        key = _canonical_task_bytes(spec)
        result = queried.get(key)
        if result is None:
            result = client.query_capacity_bound(spec)
            queried[key] = result
        self.assertEqual(result.phase, "decode")
        self.assertEqual(result.op_kind, spec["op_kind"])
        tp_rank = result.normalized_task["tp_slice"]["tensor_parallel_rank"]
        self.assertEqual(result.normalized_task["tp_slice"]["tensor_parallel"],
                         len(self.d_group.ranks))
        self.assertEqual(self.d_group.ranks[tp_rank], rank)
        bound = d_bound_by_rank_op[(rank, result.op_kind)]
        self.assertLessEqual(
            result.hbm_workspace_upper_bound_bytes,
            bound["hbm_workspace_upper_bound_bytes"],
            msg=(f"actual decode task exceeded complete manifest union: "
                 f"rank={rank} op={result.op_kind} "
                 f"actual={result.hbm_workspace_upper_bound_bytes} "
                 f"union={bound['hbm_workspace_upper_bound_bytes']}"),
        )

    def test_certificate_matches_real_prefill_emission_and_normalized_rank_shapes(self):
        certificate = self.certificate
        payload = json.loads(certificate.canonical_json)
        self.assertEqual(payload["schema"], "lru_manifest_workspace_certificate_v1")
        self.assertEqual(payload["coverage"],
                         "serialized_current_prefill_and_decode_subset_v1")
        self.assertEqual(
            certificate.fingerprint,
            __import__("hashlib").sha256(
                certificate.canonical_json.encode("utf-8")).hexdigest(),
        )
        self.assertEqual(payload["hardware_fingerprint"],
                         certificate.hardware_fingerprint)
        self.assertEqual(set(certificate.bytes_by_rank), set(range(self.config.npus_count)))
        with self.assertRaises(TypeError):
            certificate.bytes_by_rank[0] = 0

        records = [row for row in payload["proof_tasks"]
                   if row["source"] == "production_current_prefill"]
        self.assertEqual(len(records), 160)
        record_by_identity = Counter(
            (row["rank"], row["normalized_task_fingerprint"],
             _canonical_task_bytes(row["normalized_task"]))
            for row in records)
        actual = self._actual_prefill_specs()
        self.assertEqual(len(actual), len(records))

        actual_identities = Counter()
        rank_fingerprints = {}
        history_attention = None
        with patch.dict(os.environ, {"DIE_COARSE_CABI_PATH": str(_LIBRARY)}):
            with self._client() as client:
                for rank, spec in actual:
                    result = client.query_capacity_bound(spec)
                    tp_rank = result.normalized_task["tp_slice"][
                        "tensor_parallel_rank"]
                    self.assertEqual(result.phase, "prefill")
                    self.assertEqual(result.op_kind, spec["op_kind"])
                    self.assertEqual(result.normalized_task["tp_slice"][
                        "tensor_parallel"], len(self.p_group.ranks))
                    self.assertEqual(self.p_group.ranks[tp_rank], rank)
                    identity = (rank, result.normalized_task_fingerprint,
                                _canonical_task_bytes(result.normalized_task))
                    actual_identities[identity] += 1
                    rank_fingerprints.setdefault(
                        (result.op_kind, _canonical_task_bytes(
                            result.normalized_task["spans"])), set()
                    ).add(result.normalized_task_fingerprint)
                    if (rank == self.p_group.ranks[0]
                            and result.op_kind == "attention_region"
                            and "manifest_a_turn1" in
                            spec["spans"][0]["request_id"]):
                        history_attention = result.normalized_task
        self.assertEqual(actual_identities, record_by_identity)
        self.assertTrue(rank_fingerprints)
        self.assertTrue(all(len(fingerprints) == len(self.p_group.ranks)
                            for fingerprints in rank_fingerprints.values()))

        # The two-token history prefill crosses the real 512-token chunk
        # boundary. These are C++-normalized positions from the emitted task.
        self.assertIsNotNone(history_attention)
        self.assertEqual(history_attention["tensor_shapes"]["m"], 700)
        self.assertEqual([span["q_tokens"] for span in history_attention["spans"]],
                         [512, 188])
        self.assertEqual([span["kv_tokens"] for span in history_attention["spans"]],
                         [578, 766])
        self.assertEqual([span["q_start_pos"] for span in history_attention["spans"]],
                         [66, 578])

    def test_complete_decode_union_covers_valid_subset_trains_and_all_offsets(self):
        payload = json.loads(self.certificate.canonical_json)
        d_records = [row for row in payload["proof_tasks"]
                     if row["source"] == "complete_manifest_decode_progressions"]
        self.assertEqual(len(d_records), 14 * len(self.d_group.ranks))
        d_bound_by_rank_op = {}
        max_d_bound_by_rank = {}
        for row in d_records:
            task = row["normalized_task"]
            rank = row["rank"]
            op = task["op_kind"]
            self.assertEqual(task["phase"], "decode")
            self.assertEqual(task["tp_slice"]["tensor_parallel"],
                             len(self.d_group.ranks))
            self.assertEqual(self.d_group.ranks[
                task["tp_slice"]["tensor_parallel_rank"]], rank)
            self.assertEqual(row["capacity_model_revision"],
                             "workspace_lifetimes_v1")
            self.assertLessEqual(row["hbm_workspace_upper_bound_bytes"],
                                 self.certificate.bytes_by_rank[rank])
            max_d_bound_by_rank[rank] = max(
                max_d_bound_by_rank.get(rank, 0),
                row["hbm_workspace_upper_bound_bytes"])
            d_bound_by_rank_op[(rank, op)] = row
        self.assertEqual(max_d_bound_by_rank,
                         {rank: self.certificate.bytes_by_rank[rank]
                          for rank in self.d_group.ranks})

        by_session = {}
        for row in self.manifest["requests"]:
            by_session.setdefault(row["session_id"], []).append(row)
        choices = []
        for session_rows in by_session.values():
            # A session contributes zero or one current turn to a real train.
            choices.append([None, *session_rows])
        cohorts = 0
        queried: dict[bytes, object] = {}
        with patch.dict(os.environ, {
            "DIE_COARSE_CABI_PATH": str(_LIBRARY),
            "SH_FIRST_TOKEN_SPLIT": "0",
            "SH_TRAIN_MAX_ITER": "8",
        }):
            scheduler, graph, state = self._scheduler_for_train_checks()
            with self._client() as client:
                for selected in product(*choices):
                    cohort = [row for row in selected if row is not None]
                    if not cohort:
                        continue
                    for consumed in product(*(
                            range(row["decode_length"]) for row in cohort)):
                        cohorts += 1
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
                            graph_plan = self._graph_train_plan(plan, state)
                            graph.emit_iteration_train(graph_plan)
                            specs = _task_nodes(graph)
                            self.assertTrue(specs)
                            for rank, spec in specs:
                                self._assert_task_covered(
                                    client, rank, spec, d_bound_by_rank_op,
                                    queried)
        self.assertEqual(cohorts, 55)
        self.assertGreater(len(queried), 0)

    def test_first_step_and_remainder_are_covered_for_uncapped_and_capped_trains(self):
        rows = [row for row in self.manifest["requests"]
                if row["history_tokens_before"] > 0]
        self.assertEqual(len(rows), 2)
        d_records = [row for row in json.loads(self.certificate.canonical_json)[
            "proof_tasks"]
            if row["source"] == "complete_manifest_decode_progressions"]
        d_bound_by_rank_op = {
            (row["rank"], row["normalized_task"]["op_kind"]): row
            for row in d_records
        }
        queried: dict[bytes, object] = {}
        with patch.dict(os.environ, {
            "DIE_COARSE_CABI_PATH": str(_LIBRARY),
            "SH_FIRST_TOKEN_SPLIT": "1",
            "SH_TRAIN_MAX_ITER": "0",
        }):
            scheduler, graph, state = self._scheduler_for_train_checks()
            with self._client() as client:
                runtimes = [_OnlineRequestRuntime(dict(row)) for row in rows]
                state.active_decode = runtimes
                state.active_decode_lookup = set(runtimes)
                for cap in (0, 2):
                    scheduler._train_max_iter = cap
                    plan = scheduler._plan_train(state)
                    split = scheduler._first_token_plan(plan, runtimes)
                    self.assertIsNotNone(split)
                    self.assertTrue(split["split"])
                    graph_plan = self._graph_train_plan(plan, state)
                    graph_plan["first_token"] = split
                    graph.begin_batch()
                    graph.emit_train_first_step(graph_plan)
                    first_specs = _task_nodes(graph)
                    graph.emit_train_remainder(graph_plan)
                    all_specs = _task_nodes(graph)
                    remainder_specs = all_specs[len(first_specs):]
                    self.assertTrue(first_specs)
                    self.assertTrue(remainder_specs)
                    for rank, spec in first_specs + remainder_specs:
                        self._assert_task_covered(
                            client, rank, spec, d_bound_by_rank_op, queried)

    def test_invalid_manifest_and_untyped_compute_families_fail_closed(self):
        malformed = json.loads(json.dumps(self.manifest))
        malformed["requests"][1]["request_id"] = malformed["requests"][0]["request_id"]
        with self.assertRaisesRegex(WorkspaceEnvelopeError, "identities must be unique"):
            derive_manifest_workspace_envelope(malformed, self.config,
                                               library_path=_LIBRARY)

        malformed = json.loads(json.dumps(self.manifest))
        malformed["requests"][0]["prefill_length"] = True
        with self.assertRaisesRegex(WorkspaceEnvelopeError, "exact uint64"):
            derive_manifest_workspace_envelope(malformed, self.config,
                                               library_path=_LIBRARY)

        malformed = json.loads(json.dumps(self.manifest))
        malformed["requests"][0]["prefill_context_tokens"] += 1
        with self.assertRaisesRegex(WorkspaceEnvelopeError, "contradicts history"):
            derive_manifest_workspace_envelope(malformed, self.config,
                                               library_path=_LIBRARY)

        malformed = json.loads(json.dumps(self.manifest))
        malformed["compute-cost-model"] = "unsupported-model"
        with self.assertRaisesRegex(ValueError, "unsupported compute-cost-model"):
            derive_manifest_workspace_envelope(malformed, self.config,
                                               library_path=_LIBRARY)

        unsupported_config = replace(self.config, trace_granularity="token_expanded")
        with self.assertRaisesRegex(WorkspaceEnvelopeError,
                                    "unsupported workspace manifest producer"):
            derive_manifest_workspace_envelope(self.manifest, unsupported_config,
                                               library_path=_LIBRARY)

        with patch.dict(os.environ, {"DIE_COARSE_CABI_PATH": str(_LIBRARY)}):
            with self._client() as client:
                self.context.begin_emission(
                    phase="prefill", tensor_parallel=len(self.p_group.ranks),
                    tensor_parallel_rank=0, span_namespace="unsupported-family",
                    pass_spans=[(1, 1)],
                )
                untyped = self.context.build_spec(
                    "local_region", layer_count=self.config.layers,
                    weight_passes=1, fusion_policy="none")
                with self.assertRaisesRegex(WorkspaceEnvelopeError,
                                            "unsupported.*local_region"):
                    client.query_capacity_bound(untyped)

    def test_required_scheduler_installs_immutable_E_without_charging_KV_and_replays(self):
        with patch.dict(os.environ, {
            "DIE_COARSE_CABI_PATH": str(_LIBRARY),
            "SH_WORKSPACE_ADMISSION": "auto",
            "SH_TRAIN_MAX_ITER": "8",
            "SH_FIRST_TOKEN_SPLIT": "0",
        }):
            mode = _workspace_admission_mode(self.config)
            self.assertEqual(mode, "required")
            graph = GraphBatchBuilder(
                self.config,
                source_release_capacity_gates=(mode == "required"),
            )
            scheduler = WscLlmOnlineScheduler(
                manifest=self.manifest,
                config=self.config,
                graph=graph,
                workspace_admission=mode,
                defensive_reply_cache=True,
            )
            certificate = scheduler.workspace_envelope_certificate
            self.assertIsNotNone(certificate)
            self.assertEqual(dict(scheduler.kv_manager.prospective_workspace_bytes_by_rank),
                             dict(certificate.bytes_by_rank))
            self.assertTrue(graph.source_release_capacity_gates)
            self.assertIsNotNone(graph._pending_remote_store_probe)

            manager = scheduler.kv_manager
            self.assertEqual(manager._external_workspace_bytes_by_rank,
                             {rank: 0 for rank in manager.rank_ids})
            self.assertEqual(manager._external_workspace_revision_by_rank,
                             {rank: 0 for rank in manager.rank_ids})
            self.assertFalse(manager._sessions)
            self.assertFalse(manager._reservations)
            self.assertFalse(manager._events)
            initial = manager.workspace_capacity_snapshot()
            for row in initial["ranks"]:
                self.assertEqual(row["workspace_revision"], 0)
                self.assertEqual(row["resident_kv_bytes"], 0)
                self.assertEqual(row["reserved_request_bytes"], 0)
                self.assertNotIn("prospective_workspace_bytes", row)

            delta = _empty_delta(manager.rank_ids)
            response = scheduler.on_decision_batch(delta)
            rows = response["workspace_capacity_snapshot"]["ranks"]
            self.assertEqual(
                {row["rank"]: row["prospective_workspace_bytes"] for row in rows},
                dict(certificate.bytes_by_rank),
            )
            self.assertTrue(all(row["workspace_revision"] == 0
                                for row in rows))
            self.assertEqual(manager._external_workspace_bytes_by_rank,
                             {rank: 0 for rank in manager.rank_ids})
            epochs = tuple(scheduler.capacity_epoch)
            repeated = scheduler.on_decision_batch(delta)
            self.assertEqual(repeated, response)
            self.assertEqual(scheduler.delivery_count, 1)
            self.assertEqual(tuple(scheduler.capacity_epoch), epochs)
            self.assertEqual(manager.workspace_capacity_snapshot(), initial)


if __name__ == "__main__":
    unittest.main()
