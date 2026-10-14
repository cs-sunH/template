#!/usr/bin/env python3
"""Replay preserved actual legacy-v1 tasks against a newly built coarse ABI."""
import argparse
import ctypes
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    lib = ctypes.CDLL(str(args.library.resolve()))
    lib.die_coarse_create_model_v2.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_uint64)]
    lib.die_coarse_create_model_v2.restype = ctypes.c_int
    lib.die_coarse_workspace_envelope_v2.argtypes = [ctypes.c_uint64, ctypes.c_char_p]
    lib.die_coarse_workspace_envelope_v2.restype = ctypes.c_void_p
    lib.die_coarse_free_result_v2.argtypes = [ctypes.c_void_p]
    lib.die_coarse_free_result_v2.restype = None
    lib.die_coarse_destroy_model_v2.argtypes = [ctypes.c_uint64]
    lib.die_coarse_destroy_model_v2.restype = ctypes.c_int
    lib.die_coarse_last_error_v2.restype = ctypes.c_char_p
    captured_path = args.evidence / "legacy-v1-envelope-probe.json"
    captured = json.loads(captured_path.read_bytes())
    hardware_path = args.evidence / "hardware.json"
    success = expected_errors = 0
    for fixture in captured["fixtures"]:
        assert hashlib.sha256(hardware_path.read_bytes()).hexdigest() in fixture["input_sha256"].values()
        model = ctypes.c_uint64()
        assert lib.die_coarse_create_model_v2(hardware_path.read_bytes(), ctypes.byref(model)) == 0
        try:
            for query in fixture["queries"]:
                ptr = lib.die_coarse_workspace_envelope_v2(
                    model.value, json.dumps(query["task"], separators=(",", ":")).encode())
                if "expected_error" in query:
                    assert not ptr
                    assert (lib.die_coarse_last_error_v2() or b"").decode() == query["expected_error"]
                    expected_errors += 1
                    continue
                assert ptr, lib.die_coarse_last_error_v2()
                try:
                    result = json.loads(ctypes.string_at(ptr))
                finally:
                    lib.die_coarse_free_result_v2(ptr)
                assert result == query["envelope"], (fixture["fixture"], query["rank"], query["op_kind"])
                success += 1
        finally:
            assert lib.die_coarse_destroy_model_v2(model.value) == 0
    assert success == 224 and expected_errors == 16
    result = {"status": "passed", "entire_response_equal": True,
              "success_count": success, "expected_errors": expected_errors,
              "captured_sha256": hashlib.sha256(captured_path.read_bytes()).hexdigest(),
              "library": str(args.library.resolve()),
              "library_sha256": hashlib.sha256(args.library.read_bytes()).hexdigest(),
              "hardware_sha256": hashlib.sha256(hardware_path.read_bytes()).hexdigest()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
