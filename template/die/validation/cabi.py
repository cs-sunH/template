"""Small ctypes wrapper for the public coarse_die_v2 C ABI.

The helper deliberately exposes only calls needed by offline validation tools.
It never computes costs itself; every estimate and runtime result comes from the
compiled public library.
"""

from __future__ import annotations

import ctypes
import json
import os
from pathlib import Path
from typing import Any


class OwnerV2(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("schema_version", ctypes.c_uint32),
        ("request_id", ctypes.c_uint64),
        ("node_id", ctypes.c_uint64),
        ("execution_id", ctypes.c_uint64),
        ("generation", ctypes.c_uint64),
    ]

    @classmethod
    def make(cls, request_id: int, node_id: int, execution_id: int,
             generation: int = 1) -> "OwnerV2":
        return cls(ctypes.sizeof(cls), 2, request_id, node_id, execution_id,
                   generation)


def find_library(explicit: str | None = None) -> Path | None:
    if explicit:
        path = Path(explicit).expanduser().resolve()
        return path if path.is_file() else None
    env = os.environ.get("COARSE_DIE_CABI")
    if env:
        path = Path(env).expanduser().resolve()
        if path.is_file():
            return path
    here = Path(__file__).resolve()
    source = here.parents[1]
    roots = [Path.cwd(), source, source.parent]
    candidates: list[Path] = []
    for root in roots:
        candidates.extend(
            [
                root / "work/build-die/lib/libcoarse_die_cabi.so",
                root / "work/build-die/libcoarse_die_cabi.so",
                root / "build/lib/libcoarse_die_cabi.so",
                root / "build/libcoarse_die_cabi.so",
                root / "cmake-build-debug/lib/libcoarse_die_cabi.so",
                root / "cmake-build-release/lib/libcoarse_die_cabi.so",
            ]
        )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


class CoarseDieLibrary:
    def __init__(self, library_path: str | Path):
        self.path = Path(library_path).resolve()
        self.lib = ctypes.CDLL(str(self.path))
        self._setup_signatures()

    def _setup_signatures(self) -> None:
        lib = self.lib
        lib.die_coarse_abi_version.argtypes = []
        lib.die_coarse_abi_version.restype = ctypes.c_uint32
        lib.die_coarse_create_model_v2.argtypes = [ctypes.c_char_p,
                                                    ctypes.POINTER(ctypes.c_uint64)]
        lib.die_coarse_create_model_v2.restype = ctypes.c_int
        lib.die_coarse_hardware_summary_v2.argtypes = [ctypes.c_uint64]
        lib.die_coarse_hardware_summary_v2.restype = ctypes.c_void_p
        lib.die_coarse_compile_batch_v2.argtypes = [ctypes.c_uint64, ctypes.c_char_p]
        lib.die_coarse_compile_batch_v2.restype = ctypes.c_void_p
        lib.die_coarse_estimate_batch_v2.argtypes = [ctypes.c_uint64, ctypes.c_char_p]
        lib.die_coarse_estimate_batch_v2.restype = ctypes.c_void_p
        lib.die_coarse_create_runtime_v2.argtypes = [ctypes.c_uint64,
                                                     ctypes.POINTER(ctypes.c_uint64)]
        lib.die_coarse_create_runtime_v2.restype = ctypes.c_int
        lib.die_coarse_submit_plan_v2.argtypes = [ctypes.c_uint64, ctypes.c_uint64,
                                                  ctypes.POINTER(OwnerV2),
                                                  ctypes.c_uint64,
                                                  ctypes.POINTER(ctypes.c_uint64)]
        lib.die_coarse_submit_plan_v2.restype = ctypes.c_int
        lib.die_coarse_submit_endpoint_v2.argtypes = [ctypes.c_uint64, ctypes.c_char_p,
                                                      ctypes.POINTER(OwnerV2),
                                                      ctypes.c_uint64,
                                                      ctypes.POINTER(ctypes.c_uint64)]
        lib.die_coarse_submit_endpoint_v2.restype = ctypes.c_int
        lib.die_coarse_advance_to_v2.argtypes = [ctypes.c_uint64, ctypes.c_uint64]
        lib.die_coarse_advance_to_v2.restype = ctypes.c_int
        lib.die_coarse_next_event_v2.argtypes = [ctypes.c_uint64]
        lib.die_coarse_next_event_v2.restype = ctypes.c_void_p
        lib.die_coarse_cancel_execution_v2.argtypes = [ctypes.c_uint64,
                                                       ctypes.c_uint64]
        lib.die_coarse_cancel_execution_v2.restype = ctypes.c_int
        lib.die_coarse_update_capacity_v2.argtypes = [ctypes.c_uint64,
                                                      ctypes.c_uint32,
                                                      ctypes.c_double,
                                                      ctypes.c_uint64]
        lib.die_coarse_update_capacity_v2.restype = ctypes.c_int
        lib.die_coarse_query_progress_v2.argtypes = [ctypes.c_uint64]
        lib.die_coarse_query_progress_v2.restype = ctypes.c_void_p
        lib.die_coarse_query_counters_v2.argtypes = [ctypes.c_uint64]
        lib.die_coarse_query_counters_v2.restype = ctypes.c_void_p
        lib.die_coarse_export_snapshot_v2.argtypes = [ctypes.c_uint64]
        lib.die_coarse_export_snapshot_v2.restype = ctypes.c_void_p
        lib.die_coarse_import_snapshot_v2.argtypes = [ctypes.c_uint64, ctypes.c_char_p,
                                                      ctypes.POINTER(ctypes.c_uint64)]
        lib.die_coarse_import_snapshot_v2.restype = ctypes.c_int
        lib.die_coarse_fork_runtime_v2.argtypes = [ctypes.c_uint64,
                                                   ctypes.POINTER(ctypes.c_uint64)]
        lib.die_coarse_fork_runtime_v2.restype = ctypes.c_int
        lib.die_coarse_estimate_on_snapshot_v2.argtypes = [
            ctypes.c_uint64, ctypes.c_uint64, ctypes.POINTER(OwnerV2), ctypes.c_char_p
        ]
        lib.die_coarse_estimate_on_snapshot_v2.restype = ctypes.c_void_p
        lib.die_coarse_release_plan_v2.argtypes = [ctypes.c_uint64]
        lib.die_coarse_release_plan_v2.restype = ctypes.c_int
        lib.die_coarse_destroy_execution_v2.argtypes = [ctypes.c_uint64]
        lib.die_coarse_destroy_execution_v2.restype = ctypes.c_int
        lib.die_coarse_destroy_runtime_v2.argtypes = [ctypes.c_uint64]
        lib.die_coarse_destroy_runtime_v2.restype = ctypes.c_int
        lib.die_coarse_destroy_model_v2.argtypes = [ctypes.c_uint64]
        lib.die_coarse_destroy_model_v2.restype = ctypes.c_int
        lib.die_coarse_free_result_v2.argtypes = [ctypes.c_void_p]
        lib.die_coarse_free_result_v2.restype = None
        lib.die_coarse_last_error_v2.argtypes = []
        lib.die_coarse_last_error_v2.restype = ctypes.c_void_p

    def _last_error(self) -> str:
        ptr = self.lib.die_coarse_last_error_v2()
        if not ptr:
            return "unknown C ABI error"
        return ctypes.string_at(ptr).decode("utf-8", errors="replace")

    def _check(self, code: int, call: str) -> None:
        if code != 0:
            raise RuntimeError(f"{call} returned status {code}: {self._last_error()}")

    def _take_json(self, ptr: int | None, call: str) -> Any:
        if not ptr:
            raise RuntimeError(f"{call} returned null: {self._last_error()}")
        try:
            raw = ctypes.string_at(ptr).decode("utf-8")
            return json.loads(raw)
        finally:
            self.lib.die_coarse_free_result_v2(ptr)

    def abi_version(self) -> int:
        return int(self.lib.die_coarse_abi_version())

    def create_model(self, config: dict[str, Any]) -> int:
        handle = ctypes.c_uint64()
        self._check(self.lib.die_coarse_create_model_v2(
            json.dumps(config, separators=(",", ":")).encode(), ctypes.byref(handle)),
            "create_model")
        return int(handle.value)

    def hardware_summary(self, model: int) -> Any:
        return self._take_json(self.lib.die_coarse_hardware_summary_v2(model),
                               "hardware_summary")

    def compile_batch(self, model: int, tasks: list[dict[str, Any]]) -> Any:
        payload = json.dumps({"specs": tasks}, separators=(",", ":")).encode()
        return self._take_json(self.lib.die_coarse_compile_batch_v2(model, payload),
                               "compile_batch")

    def estimate_batch(self, model: int, tasks: list[dict[str, Any]]) -> Any:
        payload = json.dumps({"specs": tasks}, separators=(",", ":")).encode()
        return self._take_json(self.lib.die_coarse_estimate_batch_v2(model, payload),
                               "estimate_batch")

    def create_runtime(self, model: int) -> int:
        handle = ctypes.c_uint64()
        self._check(self.lib.die_coarse_create_runtime_v2(model, ctypes.byref(handle)),
                    "create_runtime")
        return int(handle.value)

    def submit_plan(self, runtime: int, plan: int, owner: OwnerV2,
                    now_ns: int = 0) -> int:
        handle = ctypes.c_uint64()
        self._check(self.lib.die_coarse_submit_plan_v2(
            runtime, plan, ctypes.byref(owner), now_ns, ctypes.byref(handle)),
            "submit_plan")
        return int(handle.value)

    def submit_endpoint(self, runtime: int, endpoint: dict[str, Any],
                        owner: OwnerV2, now_ns: int = 0) -> int:
        handle = ctypes.c_uint64()
        payload = json.dumps(endpoint, separators=(",", ":")).encode()
        self._check(self.lib.die_coarse_submit_endpoint_v2(
            runtime, payload, ctypes.byref(owner), now_ns, ctypes.byref(handle)),
            "submit_endpoint")
        return int(handle.value)

    def next_event(self, runtime: int) -> int | None:
        result = self._take_json(self.lib.die_coarse_next_event_v2(runtime), "next_event")
        if result is None:
            return None
        if isinstance(result, dict):
            result = result.get("next_event_ns", result.get("tick_ns"))
        return None if result is None else int(result)

    def advance_to(self, runtime: int, tick_ns: int) -> None:
        self._check(self.lib.die_coarse_advance_to_v2(runtime, tick_ns), "advance_to")

    def query_progress(self, runtime: int) -> Any:
        return self._take_json(self.lib.die_coarse_query_progress_v2(runtime),
                               "query_progress")

    def query_counters(self, runtime: int) -> Any:
        return self._take_json(self.lib.die_coarse_query_counters_v2(runtime),
                               "query_counters")

    def export_snapshot(self, runtime: int) -> Any:
        return self._take_json(self.lib.die_coarse_export_snapshot_v2(runtime),
                               "export_snapshot")

    def release_plan(self, handle: int) -> None:
        self._check(self.lib.die_coarse_release_plan_v2(handle), "release_plan")

    def destroy_execution(self, handle: int) -> None:
        self._check(self.lib.die_coarse_destroy_execution_v2(handle),
                    "destroy_execution")

    def destroy_runtime(self, handle: int) -> None:
        self._check(self.lib.die_coarse_destroy_runtime_v2(handle), "destroy_runtime")

    def destroy_model(self, handle: int) -> None:
        self._check(self.lib.die_coarse_destroy_model_v2(handle), "destroy_model")

