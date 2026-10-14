#!/usr/bin/env python3
"""FACE workspace bridge: real KV/policy hooks, wire gates and typed emitter."""
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ONLINE = Path(__file__).resolve().parent
for path in (ONLINE, ONLINE.parent):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from online.decision_bridge import BridgeError, BridgeServer
from online.online_scheduler_base import OnlineSchedulerBase
from online.online_service import _workspace_admission_mode
from online.face_online_scheduler import FaceOnlineScheduler
from online import test_admission_liveness as live
from online import test_graph_batch_builder as emission
import generate_trace


def state(manager, revision=0, active=None):
    active = active or {}
    return {"protocol_version": 1, "ranks": [
        {"rank": rank, "workspace_revision": (revision.get(rank, 0)
                                    if isinstance(revision, dict) else revision),
         "active_workspace_bytes": active.get(rank, 0)}
        for rank in manager.rank_ids]}


def delta(manager, seq=0, revision=0, active=None):
    return {"schema_version": 2, "delivery_sequence": seq,
            "delivery_epoch": seq, "tick": seq, "deferred_from_tick": seq,
            "reasons": ["TICK"], "arrivals": [], "completed_groups": [],
            "completed_nodes": [], "affected_ranks": [],
            "snapshot_handle": {"epoch": seq, "tick": seq},
            "workspace_state": state(manager, revision, active)}


class GateScheduler(OnlineSchedulerBase):
    _apply_delivery_extension = FaceOnlineScheduler._apply_delivery_extension
    _attach_response_extension = FaceOnlineScheduler._attach_response_extension
    _note_capacity_change = FaceOnlineScheduler._note_capacity_change

    def run_variant_policy(self, delta):
        self.policy_calls += 1
        self.policy_view = self.kv_manager.workspace_capacity_snapshot()


def gate_scheduler():
    # The real manager/topology comes from the established FACE liveness fixture.
    manager = live._make_scheduler().kv_manager
    scheduler = GateScheduler(manifest={"requests": [
        {"request_id": "pending", "session_id": "session"}]},
        config=SimpleNamespace(), workspace_admission="required")
    scheduler.kv_manager = manager
    scheduler._workspace_rank_to_instance = {
        rank: instance.index for instance in manager.topology.instances
        for rank in instance.ranks}
    scheduler.capacity_epoch = [0] * len(manager.topology.instances)
    scheduler.policy_calls = 0
    return scheduler


class WorkspaceDeliveryTest(unittest.TestCase):
    def setUp(self):
        from online import face_online_scheduler
        old = face_online_scheduler._P_CHUNK_HOLDER[0]
        self.addCleanup(face_online_scheduler._P_CHUNK_HOLDER.__setitem__, 0, old)

    def test_policy_sees_absolute_view_and_replay_does_not_reapply(self):
        scheduler = gate_scheduler()
        manager = scheduler.kv_manager
        before = manager.hbm_snapshots()
        request = delta(manager, revision={0: 1}, active={0: 1})
        batch = scheduler.on_decision_batch(request)
        self.assertEqual(batch["schema_version"], 2)
        self.assertEqual(batch["workspace_capacity_snapshot"], scheduler.policy_view)
        self.assertEqual(scheduler.capacity_epoch, [1, 0, 0])
        self.assertEqual(manager.hbm_snapshots(), before)
        self.assertEqual(manager.workspace_fit_remaining_bytes(0),
                         before[0].remaining_bytes - 1)
        self.assertEqual(scheduler.on_decision_batch(copy.deepcopy(request)), batch)
        self.assertEqual(scheduler.policy_calls, 1)
        self.assertEqual(scheduler.capacity_epoch, [1, 0, 0])
        scheduler.on_commit_ack({"schema_version": 2, "batch_id": 0,
                                 "delivery_sequence": 0, "success": True})
        self.assertEqual(scheduler.ack_count, 1)
        next_batch = scheduler.on_decision_batch(delta(manager, 1, revision={0: 2}))
        self.assertEqual(scheduler.capacity_epoch, [2, 0, 0])
        self.assertEqual(manager.workspace_fit_remaining_bytes(0),
                         before[0].remaining_bytes)
        self.assertEqual(next_batch["workspace_capacity_snapshot"]["ranks"][0]
                         ["workspace_revision"], 2)

    def test_bad_full_rank_envelopes_reject_before_policy_and_delivery_mutation(self):
        scheduler = gate_scheduler()
        manager = scheduler.kv_manager
        base = delta(manager)
        cases = []
        for value in (True, 1.0, -1, 1 << 64):
            malformed = copy.deepcopy(base)
            malformed["workspace_state"]["ranks"][0]["active_workspace_bytes"] = value
            cases.append(malformed)
        malformed = copy.deepcopy(base)
        malformed["workspace_state"]["ranks"].pop()
        cases.append(malformed)
        malformed = copy.deepcopy(base)
        malformed["workspace_state"]["ranks"].reverse()
        cases.append(malformed)
        malformed = copy.deepcopy(base)
        malformed["workspace_state"]["protocol_version"] = True
        cases.append(malformed)
        for field in ("delivery_sequence", "delivery_epoch", "tick",
                      "deferred_from_tick"):
            malformed = copy.deepcopy(base)
            malformed[field] = 0.0
            cases.append(malformed)
        malformed = copy.deepcopy(base)
        malformed["snapshot_handle"]["tick"] = False
        cases.append(malformed)
        for request in cases:
            with self.subTest(request=request):
                before = manager.workspace_capacity_snapshot()
                with self.assertRaises(ValueError):
                    scheduler.on_decision_batch(request)
                self.assertEqual(manager.workspace_capacity_snapshot(), before)
                self.assertEqual(scheduler.policy_calls, 0)
                self.assertEqual(scheduler.delivery_count, 0)
                self.assertEqual(scheduler.last_applied_sequence, -1)
                self.assertFalse(scheduler._emitted_by_delivery)

    def test_real_prefill_and_readonly_selection_retry_after_workspace_release(self):
        scheduler = live._make_scheduler()
        scheduler.workspace_admission = "required"
        manager = scheduler.kv_manager
        scheduler._workspace_rank_to_instance = {
            rank: instance.index for instance in manager.topology.instances
            for rank in instance.ranks}
        runtime = live._make_runtime("retry", "retry-session",
                                     prefill_context_tokens=1)
        runtime.prefill_instance_index = 0
        headroom = manager.hbm_snapshots(0)[0].remaining_bytes
        held = {rank: headroom for rank in manager.topology.instances[0].ranks}
        scheduler._apply_delivery_extension(delta(manager, revision=1, active=held))
        before = manager.hbm_snapshots()
        self.assertFalse(scheduler._probe_instance_capacity(0, (1, 1)))
        self.assertEqual(manager.hbm_snapshots(), before)
        self.assertFalse(scheduler._try_admit_prefill(runtime, 1))
        self.assertFalse(runtime.admitted_prefill)
        self.assertIn(0, scheduler.prefill_admission_dirty)
        epoch = scheduler.capacity_epoch[0]
        scheduler._apply_delivery_extension(delta(manager, 1, revision=2))
        self.assertEqual(scheduler.capacity_epoch[0], epoch + 1)
        self.assertTrue(scheduler._probe_instance_capacity(0, (1, 1)))
        self.assertTrue(scheduler._try_admit_prefill(runtime, 2))
        self.assertTrue(runtime.admitted_prefill)


class BridgeVersionTest(unittest.TestCase):
    def server(self, mode):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        server = BridgeServer(temporary.name, workspace_admission=mode)
        self.addCleanup(server._close_audit_streams)
        return server, Path(temporary.name)

    def test_v1_shape_stays_legacy_and_workspace_is_required_for_v2(self):
        for mode in ("off", "required"):
            server, directory = self.server(mode)
            request = {"schema_version": server.schema_version, "delivery_sequence": 0}
            response = {"batch_id": 0}
            if mode == "required":
                request["workspace_state"] = {"protocol_version": 1, "ranks": []}
                response["workspace_capacity_snapshot"] = {"protocol_version": 1, "ranks": []}
            (directory / 'request_0.json').write_text(json.dumps(request))
            server._handle_request(0, lambda _: response)
            saved = json.loads((directory / 'response_0.json').read_text())
            self.assertEqual(saved["schema_version"], server.schema_version)
            self.assertEqual("workspace_capacity_snapshot" in saved, mode == "required")

    def test_v2_batch_identity_is_explicit_and_exact_before_publication(self):
        for fields in ({}, {"batch_id": 1}, {"batch_id": True}, {"batch_id": 0.0}):
            with self.subTest(fields=fields):
                server, directory = self.server("required")
                (directory / 'request_0.json').write_text(json.dumps({
                    "schema_version": 2, "delivery_sequence": 0,
                    "workspace_state": {}}))
                response = {"workspace_capacity_snapshot": {
                    "protocol_version": 1, "ranks": []}, **fields}
                with self.assertRaisesRegex(BridgeError, 'batch_id'):
                    server._handle_request(0, lambda _: response)
                self.assertFalse((directory / 'response_0.json').exists())
                self.assertEqual(server._request_sequences.next_seq, 0)

    def test_v2_bad_response_or_ack_never_publishes_or_finalizes(self):
        for broken in ({}, {"schema_version": 1}, {"batch_id": 1}):
            server, directory = self.server("required")
            (directory / 'request_0.json').write_text(json.dumps({
                "schema_version": 2, "delivery_sequence": 0, "workspace_state": {}}))
            with self.assertRaises(BridgeError):
                server._handle_request(0, lambda _: dict(broken))
            self.assertFalse((directory / 'response_0.json').exists())
            self.assertEqual(server._request_sequences.next_seq, 0)
        for success in (None, False, 1):
            server, directory = self.server("required")
            ack = {"schema_version": 2, "delivery_sequence": 0, "batch_id": 0}
            if success is not None:
                ack["success"] = success
            (directory / 'commit_ack_0.json').write_text(json.dumps(ack))
            called = []
            with self.assertRaises(BridgeError):
                server._handle_ack(0, called.append)
            self.assertFalse(called)
            self.assertEqual(server._ack_sequences.next_seq, 0)

    def test_service_profile_resolves_required_and_rejects_explicit_off(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'system.json'
            config = SimpleNamespace(system_config=path)
            for profile, expected in (("coarse_die_v2", "required"),
                                      ("mesh_die_v1", "off"), (None, "off")):
                path.write_text(json.dumps({"die-model-profile": profile}))
                with patch.dict(os.environ, {"SH_WORKSPACE_ADMISSION": "auto"}):
                    self.assertEqual(_workspace_admission_mode(config), expected)
            path.write_text(json.dumps({"die-model-profile": "coarse_die_v2"}))
            with patch.dict(os.environ, {"SH_WORKSPACE_ADMISSION": "off"}):
                with self.assertRaisesRegex(ValueError, 'requires workspace'):
                    _workspace_admission_mode(config)


class CoarseEmitterTest(unittest.TestCase):
    def test_real_face_train_emits_comp_specs_accepted_by_actual_coarse_abi(self):
        library = os.environ.get("DIE_COARSE_CABI_PATH")
        if not library:
            self.skipTest("set DIE_COARSE_CABI_PATH to exercise the actual public ABI")
        from online.die_model_adapter import DieModelContext
        from online.die_model_oracle import DieModelOracle
        from online.graph_batch_builder import GraphBatchBuilder
        config_path = generate_trace.PROJECT_ROOT / 'extern/die_model/coarse/configs/example_hardware.json'
        with tempfile.TemporaryDirectory() as temporary:
            system = Path(temporary) / 'system.json'
            system.write_text(json.dumps({"die-model-profile": "coarse_die_v2"}))
            config = emission._make_config()
            config.system_config = system
            context = DieModelContext.from_service({"compute-cost-model": "die",
                "die-model-config": str(config_path.resolve())}, config)
            builder = GraphBatchBuilder(config)
            builder.set_die_context(context)
            builder.begin_batch()
            builder.emit_iteration_train(emission._train_plan(
                "batch_train_i1_1", [(1, 1), (1, 2)], iterations=2))
            body = [node for node in emission._rank_nodes(builder, 2)
                    if node['type'] == generate_trace.COMP_NODE
                    and 'die_op_spec' in node.get('compute', {})]
            selected = {node['compute']['die_op_spec']['op_kind']: node
                        for node in body}
            self.assertTrue({'attention_region', 'ffn_region'} <= set(selected))
            specs = [selected[kind]['compute']['die_op_spec']
                     for kind in ('attention_region', 'ffn_region')]
            for node in selected.values():
                self.assertEqual(node['compute']['runtime_ns'], 0)
            with DieModelOracle(lib_path=library, config_json_path=str(config_path)) as oracle:
                from face_scheduler import die_derived_face_hardware
                derived = die_derived_face_hardware(live._hardware(), oracle)
                self.assertGreater(derived.peak_perf_tflops, 0)
                self.assertGreater(derived.local_hbm_bandwidth_gbps, 0)
                from config_resolver import ResolvedHardware
                from generate_face_trace import _to_face_hardware
                resolved = ResolvedHardware(
                    source_path=config_path, capacity_profile="typed-test",
                    slug="typed-test", label="typed-test",
                    mesh_rows=derived.mesh_rows, mesh_cols=derived.mesh_cols,
                    topology_by_network_dimension=("Ring", "Ring"),
                    local_hbm_capacity_bytes=derived.local_hbm_capacity_bytes,
                    local_hbm_bandwidth_gbps=derived.local_hbm_bandwidth_gbps,
                    local_hbm_latency_ns=derived.local_hbm_latency_ns,
                    d2d_bandwidth_gbps=derived.d2d_bandwidth_gbps,
                    d2d_latency_ns=derived.d2d_latency_ns,
                    peak_perf_tflops=None, die_summary=oracle.hardware_summary(),
                    coarse_die_config=json.loads(config_path.read_text()))
                materialized = _to_face_hardware(resolved)
                self.assertEqual(materialized.peak_perf_tflops, derived.peak_perf_tflops)
                self.assertEqual(materialized.local_hbm_bandwidth_gbps,
                                 derived.local_hbm_bandwidth_gbps)
                self.assertIsNone(resolved.peak_perf_tflops)
                results = oracle.estimate(specs)
                self.assertEqual(oracle.last_errors, [None, None])
                self.assertTrue(all(result['estimated_latency_ns'] > 0 for result in results))
            output_dir = os.environ.get('FACE_WORKSPACE_EVIDENCE_DIR')
            if output_dir:
                output = Path(output_dir)
                output.mkdir(parents=True, exist_ok=True)
                (output / 'actual-face-emitter.json').write_text(json.dumps({
                    'specs': specs, 'results': results, 'nodes': [
                        selected[kind] for kind in ('attention_region', 'ffn_region')]}, indent=2)+'\n')


if __name__ == '__main__':
    unittest.main()
