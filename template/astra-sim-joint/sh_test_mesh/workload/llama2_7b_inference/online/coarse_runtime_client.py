#!/usr/bin/env python3
"""coarse_runtime_client.py -- ctypes client for the coarse-die *runtime* C ABI.

The model-level ABI (`die_coarse_create_model_v2` / `estimate_batch_v2` /
`hardware_summary_v2`) is already consumed by ``die_model_oracle``.  The
*runtime* ABI -- the one that owns execution, advances simulated time, installs
resource capacities and reports served work -- was exported by
``extern/die_model/coarse/src/Capi.cc`` but had no Python consumer.

This module closes that gap so the thermal loop can both:

* **read real activity** (cumulative served work -> incremental windows, plan
  §6.2), and
* **install a thermal cap** (`die_coarse_update_capacity_v2`, plan §9.2).

Everything is fail-closed: a missing symbol, a non-zero status or an error
string from the C ABI raises rather than silently continuing, because a thermal
run that cannot install its cap must not report a closed loop.
"""
from __future__ import annotations

import ctypes
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional


class CoarseRuntimeError(RuntimeError):
    """A runtime ABI call failed or returned an invalid payload."""


# Resource ids are the coarse-die resource table order (Hardware.cc
# `resources`/`resourceJson`).  COMPUTE_GATE is the normalized compute stage the
# thermal policy de-rates first (the gate peak is 1.0 second_of_service/s).
RESOURCE_COMPUTE_GATE = 0
RESOURCE_HBM0 = 4
RESOURCE_HBM1 = 5
RESOURCE_IDS = {
    "COMPUTE_GATE": 0, "SRAM_READ": 1, "SRAM_WRITE": 2, "LOCAL": 3,
    "HBM0": 4, "HBM1": 5, "INJECT0": 6, "INJECT1": 7,
    "XPOS": 8, "XNEG": 9, "YPOS": 10, "YNEG": 11, "HOP": 12,
}
COMPUTE_GATE_PEAK_RATE = 1.0  # second_of_service / s

# Cumulative counter categories the sampler consumes (Execution.cc
# kServedCategoryNames).  `comp` is the only category carrying COMP work.
SERVED_CATEGORIES = ("comp", "comm_send", "comm_recv", "restore", "pool", "die")
HBM_DIRECTIONS = ("read", "write")


def _decode(pointer: Optional[int], lib: Any) -> Any:
    if not pointer:
        raw = lib.die_coarse_last_error_v2()
        detail = raw.decode("utf-8", "replace") if raw else ""
        raise CoarseRuntimeError("coarse runtime ABI returned null: " + detail)
    try:
        return json.loads(ctypes.string_at(pointer).decode("utf-8"))
    finally:
        lib.die_coarse_free_result_v2(pointer)


def _guarded(lib: Any, status: int, action: str) -> None:
    if status != 0:
        raw = lib.die_coarse_last_error_v2()
        detail = raw.decode("utf-8", "replace") if raw else ""
        raise CoarseRuntimeError(f"{action} failed (status {status}): {detail}")


class CoarseRuntimeClient:
    """Thin, fail-closed wrapper around one coarse-die runtime handle.

    ``library`` may be injected (tests use a fake implementing the same
    symbols); when omitted, the shared object is located through the same
    discovery order as ``config_resolver`` (env ``DIE_COARSE_CABI_PATH`` then the
    repo build tree).
    """

    def __init__(self, runtime_handle: int, library: Any):
        self._lib = library
        self._handle = int(runtime_handle)

    # ------------------------------------------------------------ discovery --

    @staticmethod
    def _load(library_path: Optional[str]) -> Any:
        candidates = []
        if library_path:
            candidates.append(Path(library_path))
        env = __import__("os").environ.get("DIE_COARSE_CABI_PATH")
        if env:
            candidates.append(Path(env))
        repo_root = Path(__file__).resolve().parents[3]
        build = repo_root / "build" / "astra_analytical" / "build_congestion_aware"
        candidates += [build / "lib" / "libcoarse_die_cabi.so",
                       build / "bin" / "libcoarse_die_cabi.so"]
        for candidate in candidates:
            if candidate.is_file():
                return ctypes.CDLL(str(candidate))
        raise CoarseRuntimeError(
            "libcoarse_die_cabi.so not found; build it or set DIE_COARSE_CABI_PATH")

    @staticmethod
    def _declare_abi(lib: Any) -> None:
        lib.die_coarse_last_error_v2.argtypes = []
        lib.die_coarse_last_error_v2.restype = ctypes.c_char_p
        lib.die_coarse_free_result_v2.argtypes = [ctypes.c_void_p]
        lib.die_coarse_free_result_v2.restype = None
        lib.die_coarse_create_runtime_v2.argtypes = [ctypes.c_uint64, ctypes.POINTER(ctypes.c_uint64)]
        lib.die_coarse_create_runtime_v2.restype = ctypes.c_int
        lib.die_coarse_advance_to_v2.argtypes = [ctypes.c_uint64, ctypes.c_uint64]
        lib.die_coarse_advance_to_v2.restype = ctypes.c_int
        lib.die_coarse_update_capacity_v2.argtypes = [
            ctypes.c_uint64, ctypes.c_uint32, ctypes.c_double, ctypes.c_uint64]
        lib.die_coarse_update_capacity_v2.restype = ctypes.c_int
        lib.die_coarse_query_counters_v2.argtypes = [ctypes.c_uint64]
        lib.die_coarse_query_counters_v2.restype = ctypes.c_void_p
        lib.die_coarse_query_progress_v2.argtypes = [ctypes.c_uint64]
        lib.die_coarse_query_progress_v2.restype = ctypes.c_void_p
        lib.die_coarse_next_event_v2.argtypes = [ctypes.c_uint64]
        lib.die_coarse_next_event_v2.restype = ctypes.c_void_p
        lib.die_coarse_destroy_runtime_v2.argtypes = [ctypes.c_uint64]
        lib.die_coarse_destroy_runtime_v2.restype = ctypes.c_int

    # ------------------------------------------------------------- lifecycle --

    @classmethod
    def create_runtime(cls, model_handle: int, *, lib: Any = None) -> "CoarseRuntimeClient":
        lib = lib if lib is not None else cls._load(None)
        cls._declare_abi(lib)
        handle = ctypes.c_uint64()
        _guarded(lib, lib.die_coarse_create_runtime_v2(
            int(model_handle), ctypes.byref(handle)),
            "die_coarse_create_runtime_v2")
        if not handle.value:
            raise CoarseRuntimeError("die_coarse_create_runtime_v2 returned a null handle")
        return cls(handle.value, lib)

    def close(self) -> None:
        if self._handle:
            _guarded(self._lib, self._lib.die_coarse_destroy_runtime_v2(self._handle),
                     "die_coarse_destroy_runtime_v2")
            self._handle = 0

    # -------------------------------------------------------------- driving --

    def advance_to(self, now_ns: int) -> None:
        _guarded(self._lib, self._lib.die_coarse_advance_to_v2(self._handle, int(now_ns)),
                 "die_coarse_advance_to_v2")

    def update_capacity(self, resource: int, rate: float, now_ns: int) -> None:
        """Install an absolute resource rate at ``now_ns`` (the real cap install)."""
        if rate < 0.0:
            raise CoarseRuntimeError(f"refusing a negative capacity rate {rate!r}")
        _guarded(self._lib, self._lib.die_coarse_update_capacity_v2(
            self._handle, int(resource), float(rate), int(now_ns)),
            "die_coarse_update_capacity_v2")

    def counters(self) -> dict[str, Any]:
        return _decode(self._lib.die_coarse_query_counters_v2(self._handle), self._lib)

    def progress(self, execution_handle: int) -> dict[str, Any]:
        return _decode(self._lib.die_coarse_query_progress_v2(int(execution_handle)), self._lib)

    def next_event(self) -> Optional[int]:
        payload = _decode(self._lib.die_coarse_next_event_v2(self._handle), self._lib)
        return payload.get("next_event_ns")


__all__ = [
    "CoarseRuntimeClient", "CoarseRuntimeError", "RESOURCE_IDS",
    "RESOURCE_COMPUTE_GATE", "RESOURCE_HBM0", "RESOURCE_HBM1",
    "COMPUTE_GATE_PEAK_RATE", "SERVED_CATEGORIES", "HBM_DIRECTIONS",
]
