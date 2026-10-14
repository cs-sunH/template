#!/usr/bin/env python3
"""Run a Python unittest module with the official ignored Chakra schema."""

from __future__ import annotations

import importlib
from pathlib import Path
import sys
import unittest


REPO = Path(__file__).resolve().parents[2]
WORKLOAD = REPO / "sh_test_mesh/workload/llama2_7b_inference"
ONLINE = WORKLOAD / "online"
OFFICIAL_PYTHON_ROOT = REPO / "work/chakra-python-root"

sys.path[:0] = [
    str(OFFICIAL_PYTHON_ROOT),
    str(WORKLOAD),
    str(ONLINE),
]

# Preload the real protoc-generated schema before generate_trace inserts the
# checkout root at sys.path[0]. This keeps the repository's routing/import path
# intact while ensuring the tests use the official ignored generated schema.
importlib.import_module(
    "extern.graph_frontend.chakra.schema.protobuf.et_def_pb2"
)

if len(sys.argv) != 2:
    raise SystemExit("usage: run_focused_suite.py <unittest-module>")

suite = unittest.defaultTestLoader.loadTestsFromName(sys.argv[1])
result = unittest.TextTestRunner(verbosity=2).run(suite)
raise SystemExit(not result.wasSuccessful())
