import os
import subprocess
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_SCRIPT = os.path.join(_HERE, "..", "..",
                       "extern", "graph_frontend", "chakra", "schema", "protobuf",
                       "verify_et_def_pb2.py")


def test_et_def_pb2_matches_pinned_protoc():
    """Anti-drift: tracked et_def_pb2.py must equal a pinned-protoc regeneration.

    protoc absent / wrong version -> EXPLICIT pytest skip (decision 6=(a): never
    a silent pass). Any drift -> hard failure.
    """
    proc = subprocess.run([sys.executable, _SCRIPT], capture_output=True, text=True)
    out = (proc.stdout + proc.stderr).strip()
    if out.startswith("SKIP"):
        pytest.skip(out)
    assert proc.returncode == 0, out
    assert out.startswith("PASS"), out
