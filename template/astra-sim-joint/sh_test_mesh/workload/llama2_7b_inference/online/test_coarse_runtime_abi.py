#!/usr/bin/env python3
"""Real coarse-die *runtime* ABI test (plan F: the cap must reach the runtime).

Proves the previously-declared-but-unexercised runtime ABI actually works against
a real ``libcoarse_die_cabi.so``: a runtime handle is created, advanced, and a
``0.5`` COMPUTE_GATE capacity is installed and observed in the runtime counters.
Skips (never passes) when the shared object is not built.

Set ``DIE_COARSE_CABI_PATH`` to the build output, or build the repo tree so the
default discovery paths exist.
"""
import ctypes
import json
import os
import sys
import unittest
from pathlib import Path

_LLAMA_DIR = Path(__file__).resolve().parents[1]
_REPO_ROOT = _LLAMA_DIR.parents[2]
for _path in (_LLAMA_DIR,):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from online.coarse_runtime_client import (  # noqa: E402
    COMPUTE_GATE_PEAK_RATE,
    RESOURCE_COMPUTE_GATE,
    SERVED_CATEGORIES,
    CoarseRuntimeClient,
    CoarseRuntimeError,
)

_HARDWARE = _REPO_ROOT / "sh_test_mesh" / "hardware" / "face_case5_config_c.json"


def _lib_path() -> Path | None:
    candidates = []
    env = os.environ.get("DIE_COARSE_CABI_PATH")
    if env:
        candidates.append(Path(env))
    build = _REPO_ROOT / "build" / "astra_analytical" / "build_congestion_aware"
    candidates += [build / "lib" / "libcoarse_die_cabi.so",
                   build / "bin" / "libcoarse_die_cabi.so"]
    return next((path for path in candidates if path.is_file()), None)


@unittest.skipUnless(_lib_path() is not None,
                     "libcoarse_die_cabi.so not built")
class CoarseRuntimeAbiTests(unittest.TestCase):
    def setUp(self):
        self.lib = ctypes.CDLL(str(_lib_path()))
        self.lib.die_coarse_create_model_v2.argtypes = [
            ctypes.c_char_p, ctypes.POINTER(ctypes.c_uint64)]
        self.lib.die_coarse_create_model_v2.restype = ctypes.c_int
        self.lib.die_coarse_destroy_model_v2.argtypes = [ctypes.c_uint64]
        self.lib.die_coarse_destroy_model_v2.restype = ctypes.c_int
        descriptor = json.loads(_HARDWARE.read_text())["coarse-die"]
        handle = ctypes.c_uint64()
        status = self.lib.die_coarse_create_model_v2(
            json.dumps(descriptor).encode("utf-8"), ctypes.byref(handle))
        if status != 0 or not handle.value:
            raw = self.lib.die_coarse_last_error_v2()
            raise CoarseRuntimeError(
                "die_coarse_create_model_v2 failed: "
                + (raw.decode("utf-8", "replace") if raw else ""))
        self.model_handle = handle.value

    def tearDown(self):
        self.lib.die_coarse_destroy_model_v2(self.model_handle)

    def test_runtime_creates_advances_and_reports_counters(self):
        client = CoarseRuntimeClient.create_runtime(self.model_handle, lib=self.lib)
        try:
            counters = client.counters()
            self.assertIsInstance(counters, dict)
            # the real payload carries the per-resource capacity vector and the
            # lifetime served totals
            self.assertIn("capacity", counters)
            self.assertIn("served_by_category", counters)
            self.assertEqual(len(counters["capacity"]), 13)
            self.assertEqual(counters["capacity"][RESOURCE_COMPUTE_GATE],
                             COMPUTE_GATE_PEAK_RATE)
            # the declared category names are the real one
            self.assertEqual(set(counters["served_by_category"]),
                             set(SERVED_CATEGORIES))
            # the idle runtime exposes the documented call shape
            self.assertIsNone(client.next_event())
            client.advance_to(1000)
            self.assertEqual(client.counters()["now_ns"], 1000)
        finally:
            client.close()

    def test_capacity_install_reaches_the_runtime(self):
        client = CoarseRuntimeClient.create_runtime(self.model_handle, lib=self.lib)
        try:
            client.advance_to(1000)
            client.update_capacity(RESOURCE_COMPUTE_GATE, 0.5, 1000)
            after = client.counters()
            self.assertEqual(after["capacity"][RESOURCE_COMPUTE_GATE], 0.5)
            self.assertEqual(after["now_ns"], 1000)
        finally:
            client.close()

    def test_negative_rate_is_refused_before_the_abi(self):
        client = CoarseRuntimeClient.create_runtime(self.model_handle, lib=self.lib)
        try:
            with self.assertRaises(CoarseRuntimeError):
                client.update_capacity(RESOURCE_COMPUTE_GATE, -0.1, 0)
        finally:
            client.close()


if __name__ == "__main__":
    unittest.main()
