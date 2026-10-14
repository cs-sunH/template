from pathlib import Path
import hashlib
import json
import sys
import unittest

root = Path.cwd()
repo = root / "astra-sim-wscllm-LRU"
schema = repo / "work/chakra-python-root"
workload = repo / "sh_test_mesh/workload/llama2_7b_inference"
sys.path.insert(0, str(schema))
sys.path.insert(0, str(workload))
sys.path.insert(0, str(repo / "sh_test_mesh/tests"))
from extern.graph_frontend.chakra.schema.protobuf import et_def_pb2
assert Path(et_def_pb2.__file__).resolve() == (
    schema / "extern/graph_frontend/chakra/schema/protobuf/et_def_pb2.py").resolve()

online_modules = ["online." + path.stem for path in sorted((workload / "online").glob("test_*.py"))]
groups = {
    "online": online_modules,
    "typed_config_scheduler": ["test_wsc_llm_typed_hardware", "test_config_resolver", "test_wsc_llm_scheduler"],
}
reports = {}
for name, modules in groups.items():
    suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(m) for m in modules)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    reports[name] = {"modules": modules, "run": result.testsRun,
                     "passed": result.testsRun - len(result.failures) - len(result.errors) - len(result.skipped),
                     "failures": len(result.failures), "errors": len(result.errors),
                     "skipped": [(str(test), reason) for test, reason in result.skipped],
                     "successful": result.wasSuccessful()}
reports["schema"] = {"path": str(Path(et_def_pb2.__file__).resolve()),
                      "sha256": hashlib.sha256(Path(et_def_pb2.__file__).read_bytes()).hexdigest()}
reports["runner_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
(root / "work/lru-workspace-20261014/final/python-checks.json").write_text(json.dumps(reports, indent=2) + "\n")
print(json.dumps(reports, indent=2))
raise SystemExit(not all(reports[name]["successful"] for name in groups))
