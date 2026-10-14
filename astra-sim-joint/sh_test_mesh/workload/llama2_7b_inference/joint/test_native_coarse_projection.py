"""Typed Python wire -> real native projection -> canonical local Runtime.

Set NATIVE_COARSE_PROJECTION_LIBRARY and DIE_COARSE_CABI_PATH to the actual
build outputs to run the integration cases. No Python latency model is used.
"""
from dataclasses import replace
import json
import os
from pathlib import Path
import unittest

from joint.die_plan_projection import DiePlanDescriptor
from joint.native_projection import (
    NativeProjectionError, NativeProjectionLibrary, NativeProjectionWorld,
    projection_input_to_wire,
)
from joint.prediction_config import PredictionParams
from joint.resource_projection import (
    BarrierSpec, CoarseDieDomainSpec, ProjectionInput,
    ResourceProjectionError, ResourceSpec, TaskSpec, WorkItem,
)


CONFIG = Path(__file__).resolve().parents[4] / "extern/die_model/coarse/configs/example_hardware.json"
TASK = {"schema": "coarse_die_task_v2", "op_kind": "norm",
        "phase": "prefill", "tensor_shapes": {"m": 128, "hidden": 512}}


class CoarseProjectionWireTest(unittest.TestCase):
    def test_legacy_wire_omits_new_fields(self):
        legacy = ProjectionInput(
            now_ns=0, resources=(), focus_work_id="legacy",
            background_works=(WorkItem("legacy", (TaskSpec(
                "legacy#0", kind="fixed", duration_ns=1),)),))
        wire = projection_input_to_wire(legacy)
        self.assertNotIn("coarse_die_domains", wire)
        self.assertNotIn("coarse_endpoint", wire["background_works"][0]["legs"][0])

    def test_scalar_endpoint_aliases_are_rejected(self):
        with self.assertRaises(ResourceProjectionError):
            TaskSpec("bad", route=("hbm:0",), bytes_amount=1,
                     coarse_endpoint={"physical_bytes": 1})

    @unittest.skipUnless(os.environ.get("NATIVE_COARSE_PROJECTION_LIBRARY")
                         and os.environ.get("DIE_COARSE_CABI_PATH"),
                         "requires explicit native and coarse build outputs")
    def test_real_native_and_cabi_same_plan(self):
        from online.die_model_oracle import DieModelOracle
        config = json.loads(CONFIG.read_text())
        library = NativeProjectionLibrary(os.environ["NATIVE_COARSE_PROJECTION_LIBRARY"])
        with DieModelOracle(config_json_path=str(CONFIG)) as oracle:
            estimate = oracle.estimate_one(TASK)
            resources = oracle.hardware_summary()["derived"]["resources"]
            raw_bns = (resources[4]["rate"] + resources[5]["rate"]) / 1e9
        descriptor = DiePlanDescriptor(
            specs=(TASK,), route=("hbm:0",), hardware_config=config)
        comp = WorkItem("comp", (TaskSpec(
            "comp#0", kind="compute", route=("hbm:0",), die_plan=descriptor),))
        domain = CoarseDieDomainSpec("hbm:0", config)
        data = ProjectionInput(
            now_ns=0, resources=(ResourceSpec("hbm:0", "hbm", raw_bns),),
            background_works=(comp,), focus_work_id="comp", coarse_die_domains=(domain,))
        with library.create_context(data) as context:
            result = context.project((), PredictionParams())
            self.assertEqual(result.predicted_settle_done_ns,
                             estimate["estimated_latency_ns"])
            self.assertFalse(result.native_disclosure["live_world_import_supported"])
            self.assertEqual(result.native_disclosure["coarse_projection_profile"],
                             "fresh_integer_local_domains_v1")
            state = context.world_operation({"operation": "inspect"})
            self.assertEqual(state["coarse_die_domains"]["hbm:0"]["counters"]["hbm_busy_ns"], 0)
            context.world_operation({"operation": "run"})
            # A candidate query and the mutable facade own separate Runtimes.
            self.assertEqual(context.project((), PredictionParams()).predicted_settle_done_ns,
                             estimate["estimated_latency_ns"])
        changed = replace(data, efficiency=replace(data.efficiency, eta={"hbm": 0.5}))
        with library.create_context(changed) as context:
            with self.assertRaises(NativeProjectionError):
                context.project((), PredictionParams())

    @unittest.skipUnless(os.environ.get("NATIVE_COARSE_PROJECTION_LIBRARY")
                         and os.environ.get("DIE_COARSE_CABI_PATH"),
                         "requires explicit native and coarse build outputs")
    def test_native_world_snapshot_roundtrip_is_atomic_and_query_isolated(self):
        from online.die_model_oracle import DieModelOracle

        config = json.loads(CONFIG.read_text())
        with DieModelOracle(config_json_path=str(CONFIG)) as oracle:
            resources = oracle.hardware_summary()["derived"]["resources"]
            raw_bns = (resources[4]["rate"] + resources[5]["rate"]) / 1e9

        descriptor = DiePlanDescriptor(
            specs=(TASK,), route=("hbm:0",), hardware_config=config)
        coarse = WorkItem("coarse", (TaskSpec(
            "coarse#0", kind="compute", route=("hbm:0",),
            die_plan=descriptor),))
        rank0 = WorkItem("rank0", (TaskSpec(
            "rank0#0", kind="fixed", duration_ns=4,
            signals_barrier="layer:0"),))
        rank1 = WorkItem("rank1", (TaskSpec(
            "rank1#0", kind="fixed", duration_ns=7,
            signals_barrier="layer:0"),))
        data = ProjectionInput(
            now_ns=0,
            resources=(ResourceSpec("hbm:0", "hbm", raw_bns),),
            background_works=(coarse, rank0, rank1),
            focus_work_id="coarse",
            barriers=(BarrierSpec("layer:0", ("rank0#0", "rank1#0")),),
            coarse_die_domains=(CoarseDieDomainSpec("hbm:0", config),))
        library = NativeProjectionLibrary(
            os.environ["NATIVE_COARSE_PROJECTION_LIBRARY"])
        source = NativeProjectionWorld(data, library=library)
        target = NativeProjectionWorld(data, library=library)
        try:
            # Candidate projections use a fresh query world and must remain
            # independent of the mutable-session snapshot imported below.
            query_before = target._context.project((), PredictionParams())

            source.reset(event_budget=4096, conservation_check=False)
            self.assertEqual(source.run(analytic_segments=True), "ok")
            source_inspect = source._operation({"operation": "inspect"})
            barrier = source_inspect["barriers"]["layer:0"]
            self.assertEqual(tuple(barrier["signallers"]),
                             ("rank0#0", "rank1#0"))
            self.assertTrue(barrier["has_passed"])
            self.assertEqual(barrier["passed_at_ns"], 7)
            self.assertEqual(source.barriers["layer:0"].spec.signallers,
                             ("rank0#0", "rank1#0"))
            self.assertEqual(source.barriers["layer:0"].passed_at, 7)

            # Exercise an actual JSON dump/parse boundary between distinct
            # native contexts, not a Python object copy of the snapshot.
            snapshot = json.loads(json.dumps(source.export_snapshot()))
            self.assertEqual(snapshot["schema"],
                             "native_coarse_projection_world_snapshot_v1")
            self.assertEqual(snapshot["controls"]["conservation_check"], False)
            self.assertEqual(snapshot["state"]["analytic_enabled"], True)
            restored_inspect = target.import_snapshot(snapshot)
            self.assertEqual(restored_inspect, source_inspect)
            self.assertEqual(target._operation({"operation": "inspect"}),
                             source_inspect)
            self.assertFalse(target._conservation_check)
            self.assertTrue(target._analytic_mode)
            restored_barrier = target.barriers["layer:0"]
            self.assertEqual(restored_barrier.spec.signallers,
                             ("rank0#0", "rank1#0"))
            self.assertEqual(restored_barrier.arrived,
                             {"rank0#0", "rank1#0"})
            self.assertEqual(restored_barrier.passed_at, 7)

            # Corrupt a late-validated barrier field so the native candidate
            # restore fails after constructing its disposable candidate.
            before_invalid_import = target.export_snapshot()
            invalid = json.loads(json.dumps(snapshot))
            invalid["state"]["barriers"]["layer:0"]["arrived"].remove(
                "rank1#0")
            with self.assertRaises(NativeProjectionError):
                target.import_snapshot(invalid)
            self.assertEqual(target._operation({"operation": "inspect"}),
                             source_inspect)
            self.assertEqual(target.export_snapshot(), before_invalid_import)

            query_after = target._context.project((), PredictionParams())
            semantic_fields = (
                "status", "predicted_service_done_ns",
                "predicted_settle_done_ns", "reason", "dependency_attribution",
                "focus_bytes", "bytes_served_total", "background_comparisons")
            before_wire = query_before.as_dict()
            after_wire = query_after.as_dict()
            self.assertEqual({key: before_wire[key] for key in semantic_fields},
                             {key: after_wire[key] for key in semantic_fields})

            # Also transfer a live coarse Runtime.  Stop after one barrier
            # signaller has arrived, with the coarse PlanChain still active,
            # then compare the imported branch against the source branch that
            # continues without any import.
            source.reset(event_budget=4096, conservation_check=True)
            target.reset(event_budget=4096, conservation_check=True)
            self.assertNotEqual(
                source.run(horizon_ns=4, analytic_segments=True), "ok")
            active_inspect = source._operation({"operation": "inspect"})
            self.assertEqual(active_inspect["now_ns"], 4)
            self.assertIsNone(active_inspect["work_done_at"]["coarse"])
            active_barrier = active_inspect["barriers"]["layer:0"]
            self.assertFalse(active_barrier["has_passed"])
            self.assertEqual(active_barrier["arrived"], ["rank0#0"])

            active_snapshot = json.loads(json.dumps(source.export_snapshot()))
            active_entries = active_snapshot["coarse_die_domains"]["hbm:0"]["entries"]
            self.assertEqual(len(active_entries), 1)
            self.assertFalse(active_entries[0]["terminal"])
            self.assertIsNotNone(active_entries[0]["handle"])
            self.assertEqual(target.import_snapshot(active_snapshot),
                             active_inspect)
            self.assertEqual(target._operation({"operation": "inspect"}),
                             active_inspect)

            # Both branches take one identical next event and then drain.
            # `source` is the no-import continuation reference.
            self.assertEqual(source._step(), target._step())
            self.assertEqual(source._operation({"operation": "inspect"}),
                             target._operation({"operation": "inspect"}))
            self.assertEqual(source.run(analytic_segments=True),
                             target.run(analytic_segments=True))
            source_terminal = source._operation({"operation": "inspect"})
            target_terminal = target._operation({"operation": "inspect"})
            self.assertEqual(source_terminal, target_terminal)
            self.assertEqual(source_terminal["status"], "ok")
            self.assertIsNotNone(source_terminal["work_done_at"]["coarse"])
            self.assertTrue(source_terminal["barriers"]["layer:0"]["has_passed"])
        finally:
            source.close()
            target.close()


if __name__ == "__main__":
    unittest.main()
