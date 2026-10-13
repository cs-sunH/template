#!/usr/bin/env python3
"""Run coarse_die_v2 standalone estimates through its public C ABI."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HARDWARE = ROOT / "configs" / "example_hardware.json"
DEFAULT_TASKS = ROOT / "configs" / "example_tasks.json"
DEFAULT_LIBRARY = ROOT.parent / "build" / "coarse_die_v2" / "lib" / "libcoarse_die_cabi.so"
STATUS_NAMES = {
    0: "DIE_OK",
    1: "DIE_INVALID_INPUT",
    2: "DIE_INVALID_HARDWARE",
    3: "DIE_MISSING_TOPOLOGY",
    4: "DIE_UNSUPPORTED",
    5: "DIE_OVERFLOW",
    6: "DIE_INCOMPATIBLE_SCHEMA",
    7: "DIE_INCOMPATIBLE_FINGERPRINT",
    8: "DIE_INVALID_TIME",
    9: "DIE_BUSY",
    10: "DIE_STALE_HANDLE",
    11: "DIE_NUMERICAL_FAILURE",
    12: "DIE_UNSUPPORTED_LEGACY_STATE",
    13: "DIE_UNSUPPORTED_OP",
    14: "DIE_UNSUPPORTED_DTYPE",
    15: "DIE_UNSUPPORTED_HEAD_MAPPING",
    16: "DIE_UNSUPPORTED_WORKSPACE",
    17: "DIE_UNSUPPORTED_MAPPING",
    18: "DIE_UNSUPPORTED_PROJECTION_STATE",
}


class CoarseDieCapi:
    def __init__(self, path: Path):
        self.lib = ctypes.CDLL(str(path))
        self.lib.die_coarse_abi_version.argtypes = []
        self.lib.die_coarse_abi_version.restype = ctypes.c_uint32
        self.lib.die_coarse_create_model_v2.argtypes = [
            ctypes.c_char_p,
            ctypes.POINTER(ctypes.c_uint64),
        ]
        self.lib.die_coarse_create_model_v2.restype = ctypes.c_int
        self.lib.die_coarse_estimate_batch_v2.argtypes = [
            ctypes.c_uint64,
            ctypes.c_char_p,
        ]
        self.lib.die_coarse_estimate_batch_v2.restype = ctypes.c_void_p
        self.lib.die_coarse_free_result_v2.argtypes = [ctypes.c_void_p]
        self.lib.die_coarse_free_result_v2.restype = None
        self.lib.die_coarse_last_error_v2.argtypes = []
        self.lib.die_coarse_last_error_v2.restype = ctypes.c_void_p
        self.lib.die_coarse_destroy_model_v2.argtypes = [ctypes.c_uint64]
        self.lib.die_coarse_destroy_model_v2.restype = ctypes.c_int

    def last_error(self) -> str:
        pointer = self.lib.die_coarse_last_error_v2()
        if not pointer:
            return "C ABI returned no error detail"
        raw = ctypes.string_at(pointer).decode("utf-8", errors="replace")
        try:
            detail = json.loads(raw)
            return json.dumps(detail, ensure_ascii=False, sort_keys=True)
        except json.JSONDecodeError:
            return raw

    def take_json(self, pointer: int | None, operation: str) -> Any:
        if not pointer:
            raise RuntimeError(f"{operation} returned null: {self.last_error()}")
        try:
            raw = ctypes.string_at(pointer).decode("utf-8")
            return json.loads(raw)
        finally:
            self.lib.die_coarse_free_result_v2(pointer)


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read JSON file {path}: {exc}") from exc


def normalize_tasks(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, dict) and "specs" in value:
        value = value["specs"]
    elif isinstance(value, dict):
        value = [value]
    if not isinstance(value, list):
        raise ValueError("tasks JSON must be one task object, an array, or an object with a specs array")
    if not all(isinstance(task, dict) for task in value):
        raise ValueError("every task in the tasks JSON must be an object")
    return value


def render_result(payload: Any, task_count: int) -> tuple[dict[str, Any], bool]:
    if not isinstance(payload, dict):
        raise RuntimeError("estimate_batch returned a non-object JSON result")
    results = payload.get("results")
    errors = payload.get("errors")
    if not isinstance(results, list) or not isinstance(errors, list):
        raise RuntimeError("estimate_batch response is missing positional results/errors arrays")
    if len(results) != task_count or len(errors) != task_count:
        raise RuntimeError("estimate_batch response count does not match the submitted task count")

    items = []
    any_error = False
    for index, (result, error) in enumerate(zip(results, errors)):
        if error is None:
            if not isinstance(result, dict):
                any_error = True
                items.append({"index": index, "status": "error", "result": result,
                              "error": {"reason": "missing result for successful item"}})
            else:
                items.append({"index": index,
                              "status": result.get("status", "ok_assumed"),
                              "status_code": 0,
                              "result": result,
                              "error": None})
        else:
            any_error = True
            status_code = error.get("status_code") if isinstance(error, dict) else None
            status = (error.get("status") if isinstance(error, dict) else None)
            if not status:
                status = STATUS_NAMES.get(status_code, f"DIE_STATUS_{status_code}")
            items.append({"index": index, "status": status,
                          "status_code": status_code, "result": None, "error": error})

    output = {
        "schema_version": payload.get("schema_version", 2),
        "abi_version": 2,
        "status": "partial_error" if any_error else "ok_assumed",
        "cache_entries": payload.get("cache_entries"),
        "cache_accounted_bytes": payload.get("cache_accounted_bytes"),
        "cache_accounting": payload.get("cache_accounting"),
        "items": items,
    }
    return output, any_error


def write_output(value: dict[str, Any], destination: str) -> None:
    rendered = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    if destination == "-":
        sys.stdout.write(rendered)
    else:
        Path(destination).write_text(rendered, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path,
                        default=Path(os.environ.get("COARSE_DIE_CABI", DEFAULT_LIBRARY)))
    parser.add_argument("--hardware", type=Path, default=DEFAULT_HARDWARE)
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    parser.add_argument("--output", default="-",
                        help="output JSON path, or '-' for stdout (default)")
    args = parser.parse_args(argv)

    model = ctypes.c_uint64(0)
    api: CoarseDieCapi | None = None
    created = False
    exit_code = 2
    try:
        if not args.library.is_file():
            raise ValueError(f"C ABI library does not exist: {args.library}")
        hardware = read_json(args.hardware)
        if not isinstance(hardware, dict):
            raise ValueError("hardware JSON must be an object")
        tasks = normalize_tasks(read_json(args.tasks))

        api = CoarseDieCapi(args.library)
        version = int(api.lib.die_coarse_abi_version())
        if version != 2:
            raise RuntimeError(f"expected coarse Die ABI v2, found ABI v{version}")

        hardware_bytes = json.dumps(hardware, ensure_ascii=False,
                                    separators=(",", ":")).encode("utf-8")
        status = api.lib.die_coarse_create_model_v2(hardware_bytes, ctypes.byref(model))
        if status != 0:
            raise RuntimeError(f"create_model returned status {status}: {api.last_error()}")
        created = True

        tasks_bytes = json.dumps({"specs": tasks}, ensure_ascii=False,
                                 separators=(",", ":")).encode("utf-8")
        raw_result = api.lib.die_coarse_estimate_batch_v2(model.value, tasks_bytes)
        payload = api.take_json(raw_result, "estimate_batch")
        report, any_error = render_result(payload, len(tasks))
        write_output(report, args.output)
        exit_code = 1 if any_error else 0
    except (OSError, ValueError, RuntimeError, TypeError) as exc:
        print(f"estimate.py: {exc}", file=sys.stderr)
    finally:
        if created and api is not None:
            status = api.lib.die_coarse_destroy_model_v2(model.value)
            if status != 0:
                print(f"estimate.py: destroy_model returned status {status}: {api.last_error()}",
                      file=sys.stderr)
                exit_code = 2
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
