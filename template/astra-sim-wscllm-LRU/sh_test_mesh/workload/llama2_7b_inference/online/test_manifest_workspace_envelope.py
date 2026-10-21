"""Actual WSCLLM loader/materializer/emitter and C++ Plan capacity coverage.

2026-10-20(方案二直接替换):旧口径断言(实发射器 prefill proof 恰 160 条、
全量 decode progression union 的 55 组合枚举等)已随旧实现一并退役——
旧口径对 1_100 长上下文 trace 推出 402GB 地板、超过 141.7GB 物理容量而
拒跑;新口径(合成 train span 超集 + train 上限)按 astra-sim-face-LRU
同名测试逐条移植为四类:proof 身份与冻结 bound、陈旧 certificate 拒绝
(validate 四重身份负例)、坏输入 fail-closed、真实发射器计划对
certificate bound 的逐条子集证明。
"""
from __future__ import annotations

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

import generate_wsc_llm_trace as trace
import plan_materializer
from online.coarse_workspace_envelope import (
    DieWorkspaceEnvelopeClient, WorkspaceEnvelopeError, _read_owned_json,
)
from online.die_model_adapter import DieModelContext
from online.graph_batch_builder import GraphBatchBuilder
from online.manifest_workspace_envelope import derive_manifest_workspace_envelope
from online.wsc_llm_online_scheduler import WscLlmOnlineScheduler

# 仓内 congestion_aware 构建产物(coarse_workspace_envelope 的仓内发现序
# 同款);环境变量显式路径优先,缺库按旧约定 Skip(外部 C 工件缺席不是
# 测试失败)。
_LIBRARY = Path(os.environ.get(
    "DIE_WORKSPACE_ENVELOPE_TEST_LIBRARY",
    os.environ.get("DIE_COARSE_CABI_PATH", str(
        Path(__file__).resolve().parents[4] /
        "build" / "astra_analytical" / "build_congestion_aware" /
        "lib" / "libcoarse_die_cabi.so"))))


def _empty_delta(ranks) -> dict:
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
    def setUpClass(cls):
        if not _LIBRARY.is_file():
            raise unittest.SkipTest(
                "real workspace-capacity C ABI is unavailable; set "
                f"DIE_WORKSPACE_ENVELOPE_TEST_LIBRARY or DIE_COARSE_CABI_PATH: "
                f"{_LIBRARY}")
        cls.temporary = tempfile.TemporaryDirectory(prefix="wscllm-capacity-certificate-")
        cls.addClassCleanup(cls.temporary.cleanup)
        root = Path(cls.temporary.name)
        sh_test = Path(__file__).resolve().parents[3]
        cls.library = _LIBRARY
        for directory in ("hardware", "system"):
            shutil.copytree(sh_test / directory, root / directory)
        descriptor = json.loads((sh_test.parent / "extern/die_model/coarse/configs/example_hardware.json").read_text())
        import config_resolver
        capacity = config_resolver.derive_coarse_die_hardware(descriptor)["derived"]["hardware_summary"]["hbm_capacity_bytes"]
        if type(capacity) not in (int, float) or int(capacity) != capacity:
            raise AssertionError("C++ returned invalid physical capacity")
        hardware = json.loads((root / "hardware/face_case5_config_c.json").read_text())
        hardware.pop("die", None)
        hardware["coarse-die"] = descriptor
        hardware["slug"] = "wscllm_certificate_test"
        hardware["local-hbm"]["capacity-profiles"] = {"certificate": {"bytes": int(capacity), "label": "C++ derived", "note": "test physical capacity"}}
        hardware_path = root / "hardware/certificate.json"
        hardware_path.write_text(json.dumps(hardware))
        template = json.loads((root / "system/llama2_7b_die_template.json").read_text())
        template["die-model-profile"] = "coarse_die_v2"
        template_path = root / "system/certificate.json"
        template_path.write_text(json.dumps(template))
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
            "request_queue_csv": str(queue), "output_dir": str(root / "unused_output"),
            "hardware_config": str(hardware_path), "local_hbm_capacity_profile": "certificate",
            "system_template": str(template_path), "trace_granularity": "request_aggregated",
            "remote_operand_loads": "false", "kv_reserve_context_tokens": "64", "position_encoding": "rope"}
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
            cls.config = trace.load_wsc_llm_trace_config(config_path)
        finally:
            trace.SH_TEST_DIR = old_root
        cls.manifest = dict(plan_materializer._resolved_backend_keys(cls.config))
        cls.manifest["requests"] = plan_materializer._derive_manifest_requests(cls.config, {})
        cls.manifest["manifest_source"] = "real-loader-derived-test-input"
        cls.context = DieModelContext.from_service(cls.manifest, cls.config)
        cls.certificates = {cap: derive_manifest_workspace_envelope(
            cls.manifest, cls.config, train_max_iter=cap, library_path=cls.library) for cap in (2, 0)}

    def test_proof_identity_frozen_bounds_and_constructor_cap_contract(self):
        certificate = self.certificates[2]
        proof = json.loads(certificate.canonical_json)
        self.assertEqual(proof["schema"], "wscllm_manifest_workspace_certificate_v1")
        self.assertEqual(proof["coverage_revision"],
                         "wscllm_train_one_prefill_all_request_decode_superset_v1")
        self.assertEqual(proof["capacity_model_revision"], "workspace_lifetimes_v2")
        # 9 等TP组 x 6 rank:tp=6 个 relative rank x 2 phase x 14 op。
        self.assertEqual(len(proof["proof_tasks"]), 6 * 2 * 14)
        for record in proof["proof_tasks"]:
            self.assertIn("coarse_die_task_v2", record["request_json"])
            self.assertEqual(json.loads(record["response_json"])["capacity_model_revision"], "workspace_lifetimes_v2")
        self.assertEqual(set(certificate.bytes_by_rank), set(range(self.config.npus_count)))
        with self.assertRaises(TypeError):
            certificate.bytes_by_rank[0] = 0
        graph = GraphBatchBuilder(self.config, source_release_capacity_gates=True)
        scheduler = WscLlmOnlineScheduler(manifest=self.manifest, config=self.config, graph=graph,
            workspace_admission="required", train_max_iter=2,
            prospective_workspace_certificate=certificate.without_proof())
        self.assertEqual(dict(scheduler.kv_manager.prospective_workspace_bytes_by_rank),
                         dict(certificate.bytes_by_rank))
        self.assertTrue(graph.source_release_capacity_gates)
        self.assertEqual(scheduler._train_max_iter, 2)
        self.assertEqual(scheduler.workspace_envelope_certificate.canonical_json, "")
        # 快照行带 certificate 地板(_attach_response_extension 注记)。
        response = scheduler.on_decision_batch(
            _empty_delta(scheduler.kv_manager.rank_ids))
        rows = response["workspace_capacity_snapshot"]["ranks"]
        self.assertEqual({row["rank"]: row["prospective_workspace_bytes"] for row in rows},
                         dict(certificate.bytes_by_rank))
        with self.assertRaisesRegex(ValueError, "cap differs"):
            WscLlmOnlineScheduler(manifest=self.manifest, config=self.config, graph=graph,
                workspace_admission="required", train_max_iter=0,
                prospective_workspace_certificate=certificate)
        with self.assertRaisesRegex(ValueError, "workspace certificate requires workspace admission"):
            WscLlmOnlineScheduler(manifest=self.manifest, config=self.config, graph=graph,
                workspace_admission="off", train_max_iter=2,
                prospective_workspace_certificate=certificate.without_proof())

    def test_required_scheduler_installs_certificate_floor_without_charging_kv_and_replays(self):
        """2.2 调度器级用例(原 required 安装用例按新口径改写):显式注入
        certificate——地板进 manager、external maps 恒 0(不充 KV)、交付
        回放幂等;manifest 不符负例在构造器 validate 拦截。"""
        certificate = self.certificates[2].without_proof()
        graph = GraphBatchBuilder(self.config,
                                  source_release_capacity_gates=True)
        scheduler = WscLlmOnlineScheduler(
            manifest=self.manifest,
            config=self.config,
            graph=graph,
            workspace_admission="required",
            train_max_iter=2,
            prospective_workspace_certificate=certificate,
        )
        self.assertEqual(scheduler.workspace_envelope_certificate.canonical_json, "",
                         "installed certificate keeps no proof payload")
        manager = scheduler.kv_manager
        self.assertEqual(dict(manager.prospective_workspace_bytes_by_rank),
                         dict(certificate.bytes_by_rank))
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
        self.assertTrue(all(row["workspace_revision"] == 0 for row in rows))
        self.assertEqual(manager._external_workspace_bytes_by_rank,
                         {rank: 0 for rank in manager.rank_ids})
        epochs = tuple(scheduler.capacity_epoch)
        repeated = scheduler.on_decision_batch(delta)
        self.assertEqual(repeated, response)
        self.assertEqual(scheduler.delivery_count, 1)
        self.assertEqual(tuple(scheduler.capacity_epoch), epochs)
        self.assertEqual(manager.workspace_capacity_snapshot(), initial)
        # slack 对账行(verify_run_end 末尾经 _reconcile_workspace_floor
        # 产出;此处无完整请求生命周期,直接驱动同一对账入口):零交付
        # 峰值下逐 rank slack == 地板。
        scheduler._reconcile_workspace_floor()
        reconciliation = scheduler.workspace_floor_reconciliation
        self.assertEqual(reconciliation["floor_bytes_by_rank"],
                         {str(rank): certificate.bytes_by_rank[rank]
                          for rank in sorted(certificate.bytes_by_rank)})
        self.assertEqual(set(reconciliation["peak_bytes_by_rank"]),
                         set(reconciliation["floor_bytes_by_rank"]))
        self.assertTrue(all(peak == 0
                            for peak in reconciliation["peak_bytes_by_rank"].values()))

        stale_manifest = copy.deepcopy(self.manifest)
        stale_manifest["requests"][0]["request_id"] += "-changed"
        with self.assertRaisesRegex(ValueError, "manifest differs"):
            WscLlmOnlineScheduler(
                manifest=stale_manifest,
                config=self.config,
                graph=graph,
                workspace_admission="required",
                train_max_iter=2,
                prospective_workspace_certificate=certificate,
            )

    def test_stale_certificate_rejects_different_manifest_model_and_hardware(self):
        from online.manifest_workspace_envelope import validate_manifest_workspace_certificate
        certificate = self.certificates[2].without_proof()
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

    def test_actual_emitter_plans_fit_capped_uncapped_history_and_split_subsets(self):
        evidence = []
        # Actual producer spans: current head, base history recompute, restored
        # current head with history, mixed decode, and first/remainder subsets.
        shapes = [("prefill", [(4, 4), (3, 7), (1, 8), (1, 7)], 2),
                  ("prefill", [(4, 4), (4, 8), (2, 10), (4, 14), (4, 18), (1, 19)], 0),
                  ("prefill", [(4, 14), (4, 18)], 2),
                  ("decode", [(1, 8), (1, 9), (1, 7), (1, 8)], 2),
                  ("decode", [(1, 20), (1, 21), (1, 22), (1, 23), (1, 24)], 0)]
        decode_index = next(
            index for index, group in enumerate(self.config.inference_groups)
            if group.phase_role == "decode")
        with DieWorkspaceEnvelopeClient(self.context.die_config_path, self.library) as client:
            lib = client._library
            lib.die_coarse_compile_batch_v2.argtypes = [ctypes.c_uint64, ctypes.c_char_p]
            lib.die_coarse_compile_batch_v2.restype = ctypes.c_void_p
            lib.die_coarse_release_plan_v2.argtypes = [ctypes.c_uint64]
            lib.die_coarse_release_plan_v2.restype = ctypes.c_int
            for phase, spans, cap in shapes:
                for part in (spans, spans[:1], spans[1:]):
                    if not part:
                        continue
                    graph = GraphBatchBuilder(self.config)
                    graph.set_die_context(DieModelContext.from_service(self.manifest, self.config))
                    graph.begin_batch()
                    mark = graph._mark()
                    plan = {"train_id": "batch_train_capacity",
                            "instance_index": decode_index, "stage": phase}
                    graph._emit_train_body(plan, part, max(1, min(len(part), cap or len(part))))
                    graph._collect(mark)
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
                            self.assertLessEqual(peak, self.certificates[cap].bytes_by_rank[node["rank"]])
                            evidence.append({"rank": node["rank"], "op": node["compute"]["die_op_spec"]["op_kind"],
                                "cap": cap, "spans": part, "plan_hash": compiled["plan_hash"], "workspace_bytes": peak})
                    finally:
                        for handle in handles:
                            self.assertEqual(lib.die_coarse_release_plan_v2(handle), 0)
        directory = os.environ.get("WSCLLM_WORKSPACE_EVIDENCE_DIR")
        if directory:
            destination = Path(directory) / "manifest-plan-coverage.json"
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(json.dumps({"selected_plan_count": len(evidence), "plans": evidence}, indent=2))


if __name__ == "__main__":
    unittest.main()
