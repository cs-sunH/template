#!/usr/bin/env python3
"""Compile exact production-emitted COMP specs from captured Online batches."""

from __future__ import annotations

import ctypes
import os
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
WORK = ROOT / "work/lru-future-workspace-20261014/e2e"
WIRE = Path(os.environ.get("SH_RUN_DIR", str(WORK / "run"))) / "wire_audit"
HARDWARE = WORK / "generated/generated/runtime_config/workspace_e2e_coarse_8rank__workspace-e2e-coarse__edge_remote_memory_pool/resolved_die_model.json"
LIBRARY = ROOT / "work/build-lru/lib/libcoarse_die_cabi.so"

lib = ctypes.CDLL(str(LIBRARY))
lib.die_coarse_create_model_v2.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_uint64)]
lib.die_coarse_create_model_v2.restype = ctypes.c_int
lib.die_coarse_compile_batch_v2.argtypes = [ctypes.c_uint64, ctypes.c_char_p]
lib.die_coarse_compile_batch_v2.restype = ctypes.c_void_p
lib.die_coarse_free_result_v2.argtypes = [ctypes.c_void_p]
lib.die_coarse_free_result_v2.restype = None
lib.die_coarse_release_plan_v2.argtypes = [ctypes.c_uint64]
lib.die_coarse_release_plan_v2.restype = ctypes.c_int
lib.die_coarse_destroy_model_v2.argtypes = [ctypes.c_uint64]
lib.die_coarse_destroy_model_v2.restype = ctypes.c_int
lib.die_coarse_last_error_v2.argtypes = []
lib.die_coarse_last_error_v2.restype = ctypes.c_char_p

responses = []
spec_rows = []
for path in sorted(WIRE.glob("response_*.json")):
    response = json.loads(path.read_text(encoding="utf-8"))
    responses.append(response)
    sequence = response["source_delivery_sequence"]
    for node in response.get("nodes", []):
        compute = node.get("compute", {})
        spec = compute.get("die_op_spec")
        if node.get("type") == 4 and compute.get("op_role") != "control_marker" and isinstance(spec, dict):
            spec_rows.append({
                "delivery_sequence": sequence,
                "node_name": node.get("name"),
                "rank": node.get("rank"),
                "op_kind": spec.get("op_kind"),
                "fusion_policy": spec.get("fusion_policy"),
                "spec": spec,
            })

model = ctypes.c_uint64()
hardware_bytes = json.dumps(json.loads(HARDWARE.read_text(encoding="utf-8")),
                             separators=(",", ":")).encode("utf-8")
status = lib.die_coarse_create_model_v2(hardware_bytes, ctypes.byref(model))
if status != 0:
    detail = lib.die_coarse_last_error_v2()
    raise SystemExit(f"create_model status={status}: {detail!r}")

items = []
handles: list[int] = []
try:
    tasks = {"specs": [item["spec"] for item in spec_rows]}
    task_bytes = json.dumps(tasks, separators=(",", ":")).encode("utf-8")
    pointer = lib.die_coarse_compile_batch_v2(model.value, task_bytes)
    if not pointer:
        detail = lib.die_coarse_last_error_v2()
        raise SystemExit(f"compile_batch returned null: {detail!r}")
    try:
        result = json.loads(ctypes.string_at(pointer).decode("utf-8"))
    finally:
        lib.die_coarse_free_result_v2(pointer)
    results = result.get("results", [])
    errors = result.get("errors", [])
    if len(results) != len(spec_rows) or len(errors) != len(spec_rows):
        raise SystemExit("compile result count differs from captured COMP count")
    for emitted, compiled, error in zip(spec_rows, results, errors):
        if error is not None or not isinstance(compiled, dict):
            items.append({**{key: emitted[key] for key in (
                "delivery_sequence", "node_name", "rank", "op_kind",
                "fusion_policy")}, "compile_error": error})
            continue
        handle = compiled.get("plan_handle")
        if isinstance(handle, int) and handle:
            handles.append(handle)
        summary = compiled.get("summary", {})
        workspace = summary.get("hbm_workspace_peak_bytes")
        items.append({**{key: emitted[key] for key in (
            "delivery_sequence", "node_name", "rank", "op_kind",
            "fusion_policy")},
            "plan_hash": compiled.get("plan_hash"),
            "hbm_workspace_peak_bytes": workspace,
            "workspace_positive": isinstance(workspace, int) and not isinstance(workspace, bool) and workspace > 0,
            "summary": summary})
finally:
    for handle in handles:
        if lib.die_coarse_release_plan_v2(handle) != 0:
            raise SystemExit(f"release_plan failed: {lib.die_coarse_last_error_v2()!r}")
    if lib.die_coarse_destroy_model_v2(model.value) != 0:
        raise SystemExit(f"destroy_model failed: {lib.die_coarse_last_error_v2()!r}")

report = {
    "captured_response_count": len(responses),
    "captured_typed_comp_spec_count": len(spec_rows),
    "compiled_plan_count": sum("plan_hash" in item for item in items),
    "positive_workspace_plan_count": sum(item.get("workspace_positive", False) for item in items),
    "items": items,
    "provenance": {
        "hardware_sha256": __import__("hashlib").sha256(HARDWARE.read_bytes()).hexdigest(),
        "cabi_sha256": __import__("hashlib").sha256(LIBRARY.read_bytes()).hexdigest(),
        "captured_specs_from": "actual production online_service response JSON retained by strace",
        "compiler": "die_coarse_compile_batch_v2",
    },
}
output = WIRE / "compiled_plans.json"
output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                  encoding="utf-8")
print(json.dumps({key: report[key] for key in (
    "captured_response_count", "captured_typed_comp_spec_count",
    "compiled_plan_count", "positive_workspace_plan_count", "provenance")},
    indent=2, sort_keys=True))
for item in items:
    print(json.dumps({key: item.get(key) for key in (
        "delivery_sequence", "node_name", "rank", "op_kind",
        "hbm_workspace_peak_bytes", "workspace_positive", "compile_error")},
        sort_keys=True))
