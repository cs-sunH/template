#!/usr/bin/env python3
"""Compile C++ coarse Plans for every old DieOpSpec candidate kind using
actual production-emitted per-phase/per-rank specs from two finite manifests.
All output is work-only. No workload or workspace formula is implemented here.
"""
from __future__ import annotations

import ctypes
import copy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LIBRARY = ROOT / "work/lru-cpp-workspace-20261014/build/aggregate/lib/libcoarse_die_cabi.so"
REPO = ROOT / "astra-sim-wscllm-LRU"
# Literal copy of DieModelContext.OP_KINDS from the production adapter. The
# compile probe deliberately avoids importing its Chakra/protobuf dependency;
# the exact task fields still come from production-emitted response specs.
OP_KINDS = frozenset({
    "norm", "qkv_rope", "qk", "scale_mask", "softmax", "av", "o_proj",
    "residual", "attention_region", "mlp_up", "mlp_activation", "mlp_down",
    "ffn_region", "logits", "local_region",
})

FIXTURES = [
    {
        "name": "official-tiny-e2e",
        "manifest": ROOT / "work/lru-online-workspace-20261014/e2e/generated/llama2_7b_wsc_llm_inference_54npus_plan_75386cf8/manifest.json",
        "wire": ROOT / "work/lru-online-workspace-20261014/e2e/run/wire_audit",
    },
    {
        "name": "larger-workspace-probe",
        "manifest": ROOT / "work/lru-online-workspace-20261014/e2e-large-workspace-probe/generated/llama2_7b_wsc_llm_inference_54npus_plan_3dbcb61d/manifest.json",
        "wire": ROOT / "work/lru-online-workspace-20261014/e2e-large-workspace-probe/run/wire_audit",
    },
]
OUT = ROOT / "work/lru-workspace-bound-20261014/compiled-candidate-plans.json"
SOURCE_INPUTS = [
    REPO / "sh_test_mesh/workload/llama2_7b_inference/online/die_model_adapter.py",
    REPO / "sh_test_mesh/workload/llama2_7b_inference/online/graph_batch_builder.py",
    REPO / "sh_test_mesh/workload/llama2_7b_inference/online/wsc_llm_online_scheduler.py",
    REPO / "sh_test_mesh/workload/llama2_7b_inference/generate_trace.py",
    REPO / "sh_test_mesh/workload/llama2_7b_inference/generate_wsc_llm_trace.py",
    REPO / "extern/die_model/coarse/src/Mapping.cc",
    REPO / "extern/die_model/coarse/src/Compiler.cc",
    REPO / "extern/die_model/coarse/src/Capi.cc",
    REPO / "extern/die_model/coarse/include/coarse_die/Plan.hh",
]

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


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def compile_fixture(item: dict) -> dict:
    manifest_path = item["manifest"]
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest.get("compute-cost-model") != "die":
        raise RuntimeError(f"{item['name']}: manifest does not select Die backend")
    requests = manifest.get("requests")
    if not isinstance(requests, list) or len(requests) != 1:
        raise RuntimeError(f"{item['name']}: expected finite one-request fixture")
    if manifest.get("selected_request_count") != 1 or manifest.get("max_same_session_span") != 1:
        raise RuntimeError(f"{item['name']}: fixture no longer has a singleton session span")
    hardware_path = Path(manifest["die-model-config"])
    if not hardware_path.is_absolute() or not hardware_path.is_file():
        raise RuntimeError(f"{item['name']}: missing absolute resolved Die hardware config")
    hardware = json.loads(hardware_path.read_bytes())
    hardware_bytes = json.dumps(hardware, separators=(",", ":")).encode()

    seeds: dict[tuple[str, int], dict] = {}
    normalized_seed_variants: dict[tuple[str, int], set[str]] = {}
    actual: list[dict] = []
    wire_paths = sorted(item["wire"].glob("response_*.json"))
    if not wire_paths:
        raise RuntimeError(f"{item['name']}: no actual online response captures")
    for wire_path in wire_paths:
        response = json.loads(wire_path.read_text())
        for node in response.get("nodes", []):
            spec = node.get("compute", {}).get("die_op_spec")
            if not isinstance(spec, dict):
                continue
            phase = spec.get("phase")
            rank = node.get("rank")
            if phase not in ("prefill", "decode") or not isinstance(rank, int):
                raise RuntimeError(f"malformed emitted spec in {wire_path}")
            key = (phase, rank)
            seeds.setdefault(key, copy.deepcopy(spec))
            normalized = {
                k: v for k, v in spec.items()
                if k not in {"op_kind", "fusion_policy"}
            }
            normalized_seed_variants.setdefault(key, set()).add(
                json.dumps(normalized, sort_keys=True, separators=(",", ":")))
            actual.append({
                "phase": phase,
                "rank": rank,
                "op_kind": spec.get("op_kind"),
                "workspace_bytes": None,
                "spec": spec,
            })
    phases = sorted({phase for phase, _ in seeds})
    if phases != ["decode", "prefill"]:
        raise RuntimeError(f"{item['name']}: expected actual prefill+decode plans, got {phases}")
    ranks = sorted({rank for _, rank in seeds})
    if ranks != list(range(len(ranks))):
        raise RuntimeError(f"{item['name']}: emitted hardware ranks are not contiguous: {ranks}")
    if any(len(variants) != 1 for variants in normalized_seed_variants.values()):
        varying = {str(k): len(v) for k, v in normalized_seed_variants.items()
                   if len(v) != 1}
        raise RuntimeError(
            f"{item['name']}: phase/rank has multiple non-op task shapes; "
            f"enumerate each shape before issuing a bound: {varying}")

    # All v1 candidate spellings accepted by the Python adapter are probed.
    # local_region remains in this set so the C++ rejection is recorded; it is
    # not a legal generated plan because typed DieOpSpec v1 carries no graph.
    candidates = sorted(OP_KINDS)
    specs: list[dict] = []
    labels: list[dict] = []
    for (phase, rank), seed in sorted(seeds.items()):
        for op_kind in candidates:
            spec = copy.deepcopy(seed)
            spec["op_kind"] = op_kind
            spec["fusion_policy"] = (
                "auto" if op_kind in {"attention_region", "ffn_region"} else "none"
            )
            specs.append(spec)
            labels.append({"phase": phase, "rank": rank, "op_kind": op_kind,
                           "span_shape": [
                               {k: s[k] for k in ("q_tokens", "kv_tokens", "q_start_pos", "kv_start_pos")}
                               for s in seed["spans"]],
                           "layer_count": seed["layer_count"],
                           "weight_passes": seed["weight_passes"]})

    compact_specs = json.dumps({"specs": specs}, separators=(",", ":")).encode()
    model = ctypes.c_uint64()
    status = lib.die_coarse_create_model_v2(hardware_bytes, ctypes.byref(model))
    if status != 0:
        raise RuntimeError(f"create model failed: {lib.die_coarse_last_error_v2()!r}")
    handles: list[int] = []
    try:
        ptr = lib.die_coarse_compile_batch_v2(model.value, compact_specs)
        if not ptr:
            raise RuntimeError(f"compile batch failed: {lib.die_coarse_last_error_v2()!r}")
        try:
            result = json.loads(ctypes.string_at(ptr).decode())
        finally:
            lib.die_coarse_free_result_v2(ptr)
        results = result.get("results", [])
        errors = result.get("errors", [])
        if len(results) != len(specs) or len(errors) != len(specs):
            raise RuntimeError("C ABI returned a count mismatch")
        rows = []
        for label, compiled, error in zip(labels, results, errors):
            row = dict(label)
            if error is not None or not isinstance(compiled, dict):
                row["compile_error"] = error
            else:
                handle = compiled.get("plan_handle")
                if isinstance(handle, int) and handle:
                    handles.append(handle)
                summary = compiled.get("summary")
                if not isinstance(summary, dict):
                    raise RuntimeError(f"missing summary for candidate {label}")
                peak = summary.get("hbm_workspace_peak_bytes")
                if isinstance(peak, bool) or not isinstance(peak, int) or peak < 0:
                    raise RuntimeError(f"invalid exact compiler workspace result: {label} {peak!r}")
                row["workspace_bytes"] = peak
                row["plan_hash"] = compiled.get("plan_hash")
                row["selected_template"] = summary.get("selected_template")
                row["loop_kind"] = summary.get("loop_kind")
            rows.append(row)
    finally:
        for handle in handles:
            if lib.die_coarse_release_plan_v2(handle) != 0:
                raise RuntimeError(f"release plan failed: {lib.die_coarse_last_error_v2()!r}")
        if lib.die_coarse_destroy_model_v2(model.value) != 0:
            raise RuntimeError(f"destroy model failed: {lib.die_coarse_last_error_v2()!r}")

    successful = [r for r in rows if "workspace_bytes" in r]
    errors = [r for r in rows if "compile_error" in r]
    actual_kinds = sorted({r["op_kind"] for r in actual})
    generated_max = {}
    for phase in phases:
        phase_actual = [r for r in actual if r["phase"] == phase]
        for rank in sorted({r["rank"] for r in phase_actual}):
            matched = [r for r in phase_actual if r["rank"] == rank]
            # The actual path emits every category for each phase/rank and
            # the compiled plan inventory is keyed by op_kind; duplicates of
            # norm/residual do not affect the max.
            generated_max[f"{phase}/rank{rank}"] = max(
                next(x["workspace_bytes"] for x in rows
                     if x["phase"] == phase and x["rank"] == rank
                     and x["op_kind"] == r["op_kind"])
                for r in matched)
    candidate_max = {}
    per_kind_max = {}
    for phase, rank in sorted(seeds):
        eligible = [r for r in successful if r["phase"] == phase and r["rank"] == rank]
        candidate_max[f"{phase}/rank{rank}"] = max(
            r["workspace_bytes"] for r in eligible)
        per_kind_max[f"{phase}/rank{rank}"] = {
            op_kind: max(r["workspace_bytes"] for r in eligible
                         if r["op_kind"] == op_kind)
            for op_kind in candidates if any(r["op_kind"] == op_kind for r in eligible)
        }
    return {
        "fixture": item["name"],
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha(manifest_bytes),
        "selected_request_count": manifest["selected_request_count"],
        "max_same_session_span": manifest["max_same_session_span"],
        "request": requests[0],
        "hardware_config_path": str(hardware_path),
        "hardware_config_sha256": sha(hardware_path.read_bytes()),
        "coarse_die_cabi_path": str(LIBRARY),
        "coarse_die_cabi_sha256": sha(LIBRARY.read_bytes()),
        "actual_response_paths": [str(p) for p in wire_paths],
        "actual_response_sha256": {str(p): sha(p.read_bytes()) for p in wire_paths},
        "actual_phase_rank_seed_count": len(seeds),
        "actual_normalized_shape_variant_counts": {
            f"{phase}/rank{rank}": len(variants)
            for (phase, rank), variants in sorted(normalized_seed_variants.items())
        },
        "actual_generated_node_spec_count": len(actual),
        "actual_generated_op_kinds": actual_kinds,
        "adapter_candidate_op_kinds": candidates,
        "candidate_compile_count": len(rows),
        "candidate_success_count": len(successful),
        "candidate_error_count": len(errors),
        "candidate_errors": errors,
        "actual_generated_shape_max_workspace_bytes": generated_max,
        "all_opkind_selected_plan_shape_max_workspace_bytes": candidate_max,
        "all_opkind_selected_plan_by_kind_workspace_bytes": per_kind_max,
        "phase_rank_shapes": [
            {"phase": phase, "hardware_rank": rank,
             "tensor_parallel_rank": seed["tp_slice"]["tensor_parallel_rank"],
             "span_records": seed["spans"],
             "layer_count": seed["layer_count"],
             "weight_passes": seed["weight_passes"],
             "model_dims": seed["model_dims"],
             "tp_slice": seed["tp_slice"]}
            for (phase, rank), seed in sorted(seeds.items())
        ],
        "compiled_candidates": rows,
        "candidate_spec_sha256": sha(compact_specs),
    }

report = {
    "purpose": "finite manifest-bound exact C++ compiler inventory; no sampled or Python-estimated workspace formula",
    "compiler_api": "die_coarse_compile_batch_v2",
    "candidate_op_kind_source": "astra-sim-wscllm-LRU/sh_test_mesh/workload/llama2_7b_inference/online/die_model_adapter.py:OP_KINDS",
    "actual_shape_source": "captured die_op_spec objects from actual production Online response JSON",
    "proof_source_sha256": {
        str(path.relative_to(ROOT)): sha(path.read_bytes()) for path in SOURCE_INPUTS
    },
    "fixtures": [compile_fixture(x) for x in FIXTURES],
}
OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
print(json.dumps({
    "output": str(OUT),
    "cabi_sha256": sha(LIBRARY.read_bytes()),
    "fixtures": [{k: fixture[k] for k in (
        "fixture", "manifest_sha256", "actual_generated_op_kinds",
        "actual_generated_node_spec_count", "candidate_compile_count",
        "candidate_success_count", "candidate_error_count",
        "actual_generated_shape_max_workspace_bytes",
        "all_opkind_selected_plan_shape_max_workspace_bytes",
        "all_opkind_selected_plan_by_kind_workspace_bytes",
        "candidate_spec_sha256")}
        for fixture in report["fixtures"]],
}, indent=2, sort_keys=True))
