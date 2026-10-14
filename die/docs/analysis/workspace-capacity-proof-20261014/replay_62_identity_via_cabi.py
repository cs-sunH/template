#!/usr/bin/env python3
"""Replay the saved 62-task Plan identity through the built production C ABI."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[4]
BASELINE = (
    ROOT
    / "die/docs/evidence/workspace-envelope-20261014/public/identity-current.json"
)
BASELINE_COMPARISON = (
    ROOT
    / "die/docs/evidence/os-foundation-20261014/public/identity-comparison.json"
)
HARDWARE = ROOT / "die/configs/example_hardware.json"
LIBRARY = Path(
    os.environ.get(
        "DIE_COARSE_CABI_PATH",
        str(ROOT / "work/build-die/lib/libcoarse_die_cabi.so"),
    )
).resolve()
SOURCE_FILES = (
    "die/docs/analysis/os-foundation-20261014/identity_driver.cc",
    "die/include/coarse_die/Capi.h",
    "die/include/coarse_die/Plan.hh",
    "die/src/Capi.cc",
    "die/src/Compiler.cc",
    "die/src/Mapping.cc",
    "die/tests/workspace_envelope_test.cc",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def take_json(lib: ctypes.CDLL, raw: int, context: str) -> dict:
    if not raw:
        raise RuntimeError(f"{context}: {lib.die_coarse_last_error_v2().decode()}")
    try:
        return json.loads(ctypes.string_at(raw))
    finally:
        lib.die_coarse_free_result_v2(raw)


def create_model(lib: ctypes.CDLL, config: dict) -> tuple[int, dict]:
    handle = ctypes.c_uint64(0)
    status = lib.die_coarse_create_model_v2(
        json.dumps(config, separators=(",", ":")).encode(), ctypes.byref(handle)
    )
    if status != 0:
        raise RuntimeError(f"create_model returned {status}: "
                           f"{lib.die_coarse_last_error_v2().decode()}")
    summary = take_json(
        lib,
        lib.die_coarse_hardware_summary_v2(handle.value),
        "hardware_summary",
    )
    return handle.value, summary


def first_difference(expected, actual, path="$" ):
    if type(expected) is not type(actual):
        return path, expected, actual
    if isinstance(expected, dict):
        if expected.keys() != actual.keys():
            return path + ".keys", sorted(expected), sorted(actual)
        for key in sorted(expected):
            diff = first_difference(expected[key], actual[key], f"{path}.{key}")
            if diff:
                return diff
        return None
    if isinstance(expected, list):
        if len(expected) != len(actual):
            return path + ".length", len(expected), len(actual)
        for index, (left, right) in enumerate(zip(expected, actual)):
            diff = first_difference(left, right, f"{path}[{index}]")
            if diff:
                return diff
        return None
    if expected != actual:
        return path, expected, actual
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    baseline_comparison = json.loads(BASELINE_COMPARISON.read_text())
    if not baseline_comparison.get("identical") or (
        baseline_comparison.get("baseline_sha256")
        != baseline_comparison.get("current_sha256")
    ):
        raise RuntimeError("historical 62-task hash/summary comparison is not passing")
    expected_raw = BASELINE.read_bytes()
    baseline_sha = hashlib.sha256(expected_raw).hexdigest()
    if baseline_sha != baseline_comparison["current_sha256"]:
        raise RuntimeError("saved 62-task identity capture SHA does not match its record")
    expected = json.loads(expected_raw)
    if len(expected) != baseline_comparison["task_count"]:
        raise RuntimeError("saved identity capture has an unexpected task count")

    lib = ctypes.CDLL(str(LIBRARY))
    handle_t = ctypes.c_uint64
    lib.die_coarse_create_model_v2.argtypes = [ctypes.c_char_p,
                                               ctypes.POINTER(handle_t)]
    lib.die_coarse_create_model_v2.restype = ctypes.c_int
    lib.die_coarse_hardware_summary_v2.argtypes = [handle_t]
    lib.die_coarse_hardware_summary_v2.restype = ctypes.c_void_p
    lib.die_coarse_compile_batch_v2.argtypes = [handle_t, ctypes.c_char_p]
    lib.die_coarse_compile_batch_v2.restype = ctypes.c_void_p
    lib.die_coarse_release_plan_v2.argtypes = [handle_t]
    lib.die_coarse_release_plan_v2.restype = ctypes.c_int
    lib.die_coarse_destroy_model_v2.argtypes = [handle_t]
    lib.die_coarse_destroy_model_v2.restype = ctypes.c_int
    lib.die_coarse_last_error_v2.argtypes = []
    lib.die_coarse_last_error_v2.restype = ctypes.c_char_p
    lib.die_coarse_free_result_v2.argtypes = [ctypes.c_void_p]
    lib.die_coarse_free_result_v2.restype = None

    base_config = json.loads(HARDWARE.read_text())
    models: dict[str, tuple[int, dict]] = {}
    for kappa in (4.0, 0.3125):
        config = json.loads(json.dumps(base_config))
        config["performance_profile"]["kappa_quant_eqops"] = kappa
        config["performance_profile"]["kappa_acc_convert_eqops"] = kappa
        handle, hw_summary = create_model(lib, config)
        fingerprint = hw_summary["hardware_fingerprint"]
        if fingerprint in models:
            raise RuntimeError(f"duplicate reconstructed hardware fingerprint {fingerprint}")
        models[fingerprint] = (handle, config)

    results = {
        "status": "passed",
        "task_count": len(expected),
        "hash_equal_count": 0,
        "summary_equal_count": 0,
        "entire_hash_and_summary_equal": False,
        "baseline_file": str(BASELINE.relative_to(ROOT)),
        "baseline_sha256": baseline_sha,
        "library_path": str(LIBRARY),
        "library_sha256": sha256(LIBRARY),
        "hardware_config_sha256": sha256(HARDWARE),
        "example_tasks_sha256": sha256(ROOT / "die/configs/example_tasks.json"),
        "hardware_configs": {},
        "source_sha256": {path: sha256(ROOT / path) for path in SOURCE_FILES},
    }
    for fingerprint, (handle, config) in models.items():
        results["hardware_configs"][fingerprint] = {
            "kappa_quant_eqops": config["performance_profile"]["kappa_quant_eqops"],
            "kappa_acc_convert_eqops": config["performance_profile"]["kappa_acc_convert_eqops"],
        }

    expected_by_hardware: dict[str, list[dict]] = {}
    for item in expected:
        expected_by_hardware.setdefault(item["hardware"], []).append(item)

    all_equal = True
    for fingerprint, items in expected_by_hardware.items():
        if fingerprint not in models:
            raise RuntimeError(f"cannot reconstruct hardware {fingerprint}")
        handle, _ = models[fingerprint]
        payload = json.dumps(
            {"specs": [item["task"] for item in items]},
            separators=(",", ":"),
        ).encode()
        response = take_json(
            lib, lib.die_coarse_compile_batch_v2(handle, payload),
            f"compile_batch {fingerprint}",
        )
        if len(response.get("results", [])) != len(items):
            raise RuntimeError(f"compile_batch result count mismatch for {fingerprint}")
        if any(error is not None for error in response.get("errors", [])):
            raise RuntimeError(f"compile_batch returned errors for {fingerprint}: "
                               f"{response['errors']}")
        for item, actual in zip(items, response["results"]):
            if actual is None:
                raise RuntimeError("compile_batch returned a null Plan result")
            hash_equal = item["hash"] == actual["plan_hash"]
            summary_equal = item["summary"] == actual["summary"]
            results["hash_equal_count"] += int(hash_equal)
            results["summary_equal_count"] += int(summary_equal)
            if not hash_equal or not summary_equal:
                all_equal = False
                results.setdefault("mismatches", []).append({
                    "op_kind": item["task"].get("op_kind"),
                    "plan_hash_equal": hash_equal,
                    "summary_equal": summary_equal,
                    "hash_expected": item["hash"],
                    "hash_actual": actual["plan_hash"],
                    "summary_first_difference": first_difference(
                        item["summary"], actual["summary"]),
                })
            if lib.die_coarse_release_plan_v2(actual["plan_handle"]) != 0:
                raise RuntimeError("release_plan failed after identity comparison")

    results["entire_hash_and_summary_equal"] = all_equal
    if not all_equal:
        results["status"] = "failed"
    for handle, _ in models.values():
        if lib.die_coarse_destroy_model_v2(handle) != 0:
            raise RuntimeError("destroy_model failed after identity replay")

    serialized = json.dumps(results, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")
    return 0 if all_equal else 1


if __name__ == "__main__":
    sys.exit(main())
