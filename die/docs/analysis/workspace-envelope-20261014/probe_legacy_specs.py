#!/usr/bin/env python3
"""Query new all-mapping envelopes for real legacy-v1 online task shapes.

No workspace formula or Plan ranking is implemented here. The comparison
uses previously captured selected-Plan results and independently audited
all-mapping maxima for the same two finite singleton manifests.
"""
import copy
import ctypes
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
BASELINE = ROOT / "work/lru-workspace-bound-20261014/compiled-candidate-plans.json"
LIBRARY = ROOT / "die/work/lru-workspace-envelope-20261014/build/lib/libcoarse_die_cabi.so"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    baseline = json.loads(BASELINE.read_bytes())
    lib = ctypes.CDLL(str(LIBRARY))
    lib.die_coarse_create_model_v2.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_uint64)]
    lib.die_coarse_create_model_v2.restype = ctypes.c_int
    lib.die_coarse_workspace_envelope_v2.argtypes = [ctypes.c_uint64, ctypes.c_char_p]
    lib.die_coarse_workspace_envelope_v2.restype = ctypes.c_void_p
    lib.die_coarse_free_result_v2.argtypes = [ctypes.c_void_p]
    lib.die_coarse_free_result_v2.restype = None
    lib.die_coarse_destroy_model_v2.argtypes = [ctypes.c_uint64]
    lib.die_coarse_destroy_model_v2.restype = ctypes.c_int
    lib.die_coarse_last_error_v2.argtypes = []
    lib.die_coarse_last_error_v2.restype = ctypes.c_char_p

    fixtures = []
    total_ok = total_expected_errors = 0
    for fixture in baseline["fixtures"]:
        inputs = {}
        seeds = {}
        variants = {}
        for path_text in fixture["actual_response_paths"]:
            path = Path(path_text)
            assert sha(path) == fixture["actual_response_sha256"][str(path)]
            inputs[str(path)] = sha(path)
            for node in json.loads(path.read_bytes()).get("nodes", []):
                spec = node.get("compute", {}).get("die_op_spec")
                if not isinstance(spec, dict):
                    continue
                key = (spec["phase"], node["rank"])
                seeds.setdefault(key, spec)
                shape = {k: v for k, v in spec.items() if k not in {"op_kind", "fusion_policy"}}
                variants.setdefault(key, set()).add(json.dumps(shape, sort_keys=True))
        assert len(seeds) == fixture["actual_phase_rank_seed_count"] == 8
        assert all(len(shapes) == 1 for shapes in variants.values())
        hardware_path = Path(fixture["hardware_config_path"])
        assert sha(hardware_path) == fixture["hardware_config_sha256"]
        inputs[str(hardware_path)] = sha(hardware_path)
        model = ctypes.c_uint64()
        assert lib.die_coarse_create_model_v2(hardware_path.read_bytes(), ctypes.byref(model)) == 0
        rows = []
        maximum = {}
        try:
            for (phase, rank), seed in sorted(seeds.items()):
                for kind in fixture["adapter_candidate_op_kinds"]:
                    spec = copy.deepcopy(seed)
                    spec["op_kind"] = kind
                    spec["fusion_policy"] = "auto" if kind in {"attention_region", "ffn_region"} else "none"
                    raw = json.dumps(spec, sort_keys=True, separators=(",", ":")).encode()
                    ptr = lib.die_coarse_workspace_envelope_v2(model.value, raw)
                    row = {"phase": phase, "rank": rank, "op_kind": kind, "task": spec,
                           "input_sha256": hashlib.sha256(raw).hexdigest()}
                    if not ptr:
                        error = (lib.die_coarse_last_error_v2() or b"").decode()
                        assert kind == "local_region" and "DieOpSpec v1 carries no typed mathematical graph" in error, error
                        row["expected_error"] = error
                        total_expected_errors += 1
                    else:
                        try:
                            result = json.loads(ctypes.string_at(ptr))
                        finally:
                            lib.die_coarse_free_result_v2(ptr)
                        assert kind != "local_region"
                        assert result["schema_version"] == 2 and result["workspace_envelope_schema_version"] == 1
                        peak = result["max_hbm_workspace_peak_bytes"]
                        assert type(peak) is int and 0 <= peak < 2**64
                        assert type(result["candidate_count"]) is int and 0 < result["candidate_count"] <= 256
                        assert result["hardware_fingerprint"] and result["normalized_task_fingerprint"]
                        assert result["op_kind"] == kind and result["phase"] == phase
                        old = next(r for r in fixture["compiled_candidates"]
                                   if (r["phase"], r["rank"], r["op_kind"]) == (phase, rank, kind))
                        assert old["workspace_bytes"] <= peak
                        row["selected_plan_workspace_bytes"] = old["workspace_bytes"]
                        row["envelope"] = result
                        maximum[f"{phase}:{rank}"] = max(maximum.get(f"{phase}:{rank}", 0), peak)
                        total_ok += 1
                    rows.append(row)
        finally:
            assert lib.die_coarse_destroy_model_v2(model.value) == 0
        expected = {"prefill": 40960, "decode": 648} if fixture["fixture"] == "official-tiny-e2e" else {"prefill": 65536, "decode": 1032}
        assert all(value == expected[key.split(":")[0]] for key, value in maximum.items()), maximum
        fixtures.append({"fixture": fixture["fixture"], "input_sha256": inputs,
                         "all_mapping_shape_max_bytes": maximum, "queries": rows})
    assert total_ok == 224 and total_expected_errors == 16
    evidence = {"status": "passed", "query_count": 240, "success_count": total_ok,
                "expected_local_region_v1_rejections": total_expected_errors,
                "library_path": str(LIBRARY), "library_sha256": sha(LIBRARY),
                "selected_plan_baseline_sha256": sha(BASELINE), "fixtures": fixtures}
    (OUT / "legacy-v1-envelope-probe.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps({k: v for k, v in evidence.items() if k != "fixtures"}, sort_keys=True))


if __name__ == "__main__":
    main()
