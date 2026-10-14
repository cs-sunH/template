from __future__ import annotations

from pathlib import Path
import sys
import unittest


REPO = Path("/workspace/template/astra-sim-wscllm-LRU")
SCHEMA_ROOT = REPO / "work/chakra-python-root"
WORKLOAD = REPO / "sh_test_mesh/workload/llama2_7b_inference"
sys.path.insert(0, str(SCHEMA_ROOT))
sys.path.insert(0, str(WORKLOAD))

# Preload the official protoc-generated binding from the ignored work tree
# before a route import causes generate_trace to put REPO at sys.path[0].
from extern.graph_frontend.chakra.schema.protobuf import et_def_pb2  # noqa: E402

expected_module = (
    SCHEMA_ROOT
    / "extern/graph_frontend/chakra/schema/protobuf/et_def_pb2.py"
).resolve()
if Path(et_def_pb2.__file__).resolve() != expected_module:
    raise RuntimeError(
        f"unexpected Chakra Python schema source: {et_def_pb2.__file__}"
    )

DEFAULT_MODULES = (
    "test_workspace_admission_manager",
    "test_wscllm_kv_incremental_invariants",
    "test_tiered_eviction_sequence",
    "test_kv_delta_journal",
    "online.test_admission_net_credit",
    "online.test_admission_eviction_accumulation",
)
MODULES = tuple(sys.argv[1:]) or DEFAULT_MODULES
loader = unittest.defaultTestLoader
suite = unittest.TestSuite()
for module_name in MODULES:
    suite.addTests(loader.loadTestsFromName(module_name))

result = unittest.TextTestRunner(verbosity=2).run(suite)
raise SystemExit(not result.wasSuccessful())
