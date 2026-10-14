#!/usr/bin/env python3
"""Compile FFN specs built by the production LRU DieModelContext/C ABI."""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import sys
from pathlib import Path


ROOT = next(p for p in Path(__file__).resolve().parents if (p / "die/src/Execution.cc").is_file())
WORK = ROOT / "work/lru-online-workspace-20261014/e2e-large-workspace-probe"
REPO = ROOT / "astra-sim-wscllm-LRU"
PLAN = next((WORK / "generated").glob(
    "llama2_7b_wsc_llm_inference_54npus_plan_*/manifest.json"))
PLAN_DIR = PLAN.parent
TRACE_CONFIG = WORK / "trace_config.csv"
HARDWARE = next((WORK / "generated/generated/runtime_config").glob(
    "workspace_e2e_coarse_8rank__workspace-e2e-coarse__edge_remote_memory_pool/"
    "resolved_die_model.json"))
HARDWARE_FIXTURE = WORK / "hardware_coarse_8rank.json"
LIBRARY = ROOT / "work/lru-cpp-workspace-20261014/build/aggregate/lib/libcoarse_die_cabi.so"
OUT_DIR = WORK / "precompile"
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(REPO / "sh_test_mesh/workload/llama2_7b_inference"))
sys.path.insert(0, str(REPO / "sh_test_mesh"))
os.environ["DIE_COARSE_CABI_PATH"] = str(LIBRARY)

import generate_wsc_llm_trace as trace  # noqa: E402
from online.die_model_adapter import DieModelContext  # noqa: E402

manifest = json.loads(PLAN.read_text(encoding="utf-8"))
config = trace.load_wsc_llm_trace_config(TRACE_CONFIG)
context = DieModelContext.from_service(manifest, config)
if context is None:
    raise SystemExit("official plan does not select the production DIE backend")

request = manifest["requests"][0]
queue_index = int(request["queue_index"])
session_id = str(request["session_id"])
turn_index = int(request["turn_index"])
prefill_span = (int(request["prefill_length"]),
                int(request["prefill_context_tokens"]))
decode_span = (int(request["decode_length"]),
               int(request["final_context_tokens"]))
prefill_namespace = (
    f"q{queue_index:04d}_{session_id}_turn{turn_index}_"
    f"{request['request_id']}_current_prefill")

specs = []
for phase, namespace, span in (
        ("prefill", prefill_namespace, prefill_span),
        ("decode", "batch_train_i1_1", decode_span)):
    for tp_rank in range(4):
        context.begin_emission(
            phase=phase,
            tensor_parallel=4,
            tensor_parallel_rank=tp_rank,
            span_namespace=namespace,
            pass_spans=[span],
        )
        specs.append({
            "phase": phase,
            "tensor_parallel_rank": tp_rank,
            "spec": context.build_spec(
                "ffn_region", layer_count=int(config.layers), weight_passes=1,
                fusion_policy="auto"),
        })

spec_path = OUT_DIR / "official_ffn_specs.json"
spec_bytes = json.dumps({"specs": [row["spec"] for row in specs]},
                        separators=(",", ":")).encode("utf-8")
# Retain the exact compact JSON payload passed to the C ABI so its hash is a
# direct identity for the compiled task, not a pretty-printed equivalent.
spec_path.write_bytes(spec_bytes)

lib = ctypes.CDLL(str(LIBRARY))
lib.die_coarse_create_model_v2.argtypes = [ctypes.c_char_p,
                                            ctypes.POINTER(ctypes.c_uint64)]
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

hardware_bytes = json.dumps(json.loads(HARDWARE.read_text(encoding="utf-8")),
                             separators=(",", ":")).encode("utf-8")
model = ctypes.c_uint64()
status = lib.die_coarse_create_model_v2(hardware_bytes, ctypes.byref(model))
if status != 0:
    raise SystemExit(f"create_model status={status}: {lib.die_coarse_last_error_v2()!r}")

handles: list[int] = []
try:
    pointer = lib.die_coarse_compile_batch_v2(model.value, spec_bytes)
    if not pointer:
        raise SystemExit(f"compile_batch returned null: {lib.die_coarse_last_error_v2()!r}")
    try:
        compiled = json.loads(ctypes.string_at(pointer).decode("utf-8"))
    finally:
        lib.die_coarse_free_result_v2(pointer)
    results = compiled.get("results", [])
    errors = compiled.get("errors", [])
    if len(results) != len(specs) or len(errors) != len(specs):
        raise SystemExit("C ABI result count differs from production FFN spec count")
    rows = []
    for source, result, error in zip(specs, results, errors):
        if error is not None or not isinstance(result, dict):
            rows.append({"phase": source["phase"],
                         "tensor_parallel_rank": source["tensor_parallel_rank"],
                         "error": error})
            continue
        handle = result.get("plan_handle")
        if isinstance(handle, int) and handle:
            handles.append(handle)
        summary = result.get("summary", {})
        rows.append({
            "phase": source["phase"],
            "tensor_parallel_rank": source["tensor_parallel_rank"],
            "plan_hash": result.get("plan_hash"),
            "hbm_workspace_peak_bytes": summary.get("hbm_workspace_peak_bytes"),
            "tk": summary.get("tk"),
            "fused_tk_capacity_probe_order": summary.get(
                "fused_tk_capacity_probe_order"),
            "executed_int4_ops": summary.get("executed_int4_ops"),
            "executed_matrix_ops": summary.get("executed_matrix_ops"),
            "vec_equivalent_ops": summary.get("vec_equivalent_ops"),
            "sram_peak_bytes_per_core": summary.get("sram_peak_bytes_per_core"),
            "hbm_read_bytes_total": summary.get("hbm_read_bytes_total"),
            "hbm_write_bytes_total": summary.get("hbm_write_bytes_total"),
            "selected_template": summary.get("selected_template"),
            "summary": summary,
        })
finally:
    for handle in handles:
        status = lib.die_coarse_release_plan_v2(handle)
        if status != 0:
            raise SystemExit(f"release_plan failed: {lib.die_coarse_last_error_v2()!r}")
    status = lib.die_coarse_destroy_model_v2(model.value)
    if status != 0:
        raise SystemExit(f"destroy_model failed: {lib.die_coarse_last_error_v2()!r}")

fixture_hardware = json.loads(HARDWARE_FIXTURE.read_text(encoding="utf-8"))
capacity_bytes = int(fixture_hardware["local-hbm"]["capacity-profiles"]
                     ["workspace-e2e-coarse"]["bytes"])
report = {
    "probe": "official LRU DieModelContext FFN specs compiled by die_coarse_compile_batch_v2",
    "plan_manifest": str(PLAN),
    "model_name": manifest.get("model", {}).get("name"),
    "model_dims": context.model_dims,
    "capacity_bytes_per_rank": capacity_bytes,
    "span_sources": {
        "prefill": {"q_tokens": prefill_span[0], "kv_tokens": prefill_span[1]},
        "decode": {"q_tokens": decode_span[0], "kv_tokens": decode_span[1]},
    },
    "spec_sha256": hashlib.sha256(spec_path.read_bytes()).hexdigest(),
    "hardware_sha256": hashlib.sha256(HARDWARE.read_bytes()).hexdigest(),
    "cabi_sha256": hashlib.sha256(LIBRARY.read_bytes()).hexdigest(),
    "results": rows,
    "positive_workspace_count": sum(
        isinstance(row.get("hbm_workspace_peak_bytes"), int) and
        row["hbm_workspace_peak_bytes"] > 0 for row in rows),
}
output = OUT_DIR / "official_ffn_precompile.json"
output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                  encoding="utf-8")
print(json.dumps({
    key: report[key] for key in
    ("probe", "plan_manifest", "model_name", "model_dims",
     "capacity_bytes_per_rank", "spec_sha256", "hardware_sha256",
     "cabi_sha256", "positive_workspace_count")},
    indent=2, sort_keys=True))
for row in rows:
    print(json.dumps({key: row.get(key) for key in
                      ("phase", "tensor_parallel_rank", "hbm_workspace_peak_bytes",
                       "tk", "fused_tk_capacity_probe_order", "selected_template",
                       "error")}, sort_keys=True))
