"""Scratch smoke: send each JOINT coarse builder op through the real C ABI."""

from __future__ import annotations

from pathlib import Path

from die_model_adapter import _COARSE_DIE_OP_KINDS, build_coarse_die_task
from die_model_oracle import DieModelOracle
from test_coarse_die_adapter import _context


ROOT = Path(__file__).resolve().parents[2]
ops = sorted(_COARSE_DIE_OP_KINDS)
spans = [{
    "request_id": "joint-v2-op-probe",
    "q_tokens": 8,
    "kv_tokens": 8,
    "q_start_pos": 0,
    "kv_start_pos": 0,
}]
tasks = [
    build_coarse_die_task(
        op,
        _context(),
        spans,
        layer_count=1,
        weight_passes=1,
        fusion_policy="auto",
        graph_identity={"request_id": "joint-v2-op-probe", "node_id": index},
    )
    for index, op in enumerate(ops)
]

from extern.graph_frontend.chakra.schema.protobuf import et_def_pb2

print("generated_protobuf_node_type:", et_def_pb2.NodeType.COMP_NODE)
config = ROOT / "astra-sim-joint/extern/die_model/coarse/configs/example_hardware.json"
with DieModelOracle(config_json_path=str(config)) as oracle:
    results = oracle.estimate(tasks)
    errors = [
        (op, oracle.last_errors[index])
        for index, op in enumerate(ops)
        if oracle.last_errors[index] is not None
    ]
    print(
        "op_kinds={}; estimates={}; errors={}".format(
            len(ops), sum(result is not None for result in results), errors
        )
    )
    for op, result in zip(ops, results):
        print(op, "ERROR" if result is None else result.get("estimated_latency_ns"))
    if errors:
        raise SystemExit(1)
