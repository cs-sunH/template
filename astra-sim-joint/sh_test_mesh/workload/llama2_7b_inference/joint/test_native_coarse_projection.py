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
    NativeProjectionError, NativeProjectionLibrary, projection_input_to_wire,
)
from joint.prediction_config import PredictionParams
from joint.resource_projection import (
    CoarseDieDomainSpec, ProjectionInput, ResourceProjectionError,
    ResourceSpec, TaskSpec, WorkItem,
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


if __name__ == "__main__":
    unittest.main()
