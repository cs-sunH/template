#!/usr/bin/env python3
"""Verify the tracked et_def_pb2.py is byte-identical to a regeneration of the
tracked et_def.proto with the PINNED protoc using the canonical repo-root
invocation (protoc -I . --python_out <tmp> extern/graph_frontend/chakra/
schema/protobuf/et_def.proto run at the repository root, so the generated
descriptor embeds the repo-relative source path). Anti-drift check mandated by
the 2026-10-17 consensus decision 6=(a). protoc absent or version-mismatched
-> EXPLICIT skip (never a silent pass). Exit codes: 0 = PASS or SKIP, 1 = FAIL."""
import hashlib, os, shutil, subprocess, sys, tempfile

PINNED = "libprotoc 3.21.12"
HERE = os.path.dirname(os.path.abspath(__file__))          # .../schema/protobuf
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
TRACKED = os.path.join(HERE, "et_def_pb2.py")
PROTO_REL = "extern/graph_frontend/chakra/schema/protobuf/et_def.proto"
OUT_REL = "extern/graph_frontend/chakra/schema/protobuf/et_def_pb2.py"

def main():
    if not os.path.exists(TRACKED):
        print("FAIL: et_def_pb2.py missing next to et_def.proto; regenerate with the pinned protoc (%s)" % PINNED)
        return 1
    if not os.path.exists(os.path.join(REPO_ROOT, PROTO_REL)):
        print("SKIP (explicit): repository root not found above schema/protobuf; layout not recognized")
        return 0
    protoc = shutil.which("protoc")
    if protoc is None:
        print("SKIP (explicit): protoc not on PATH; cannot verify et_def_pb2.py against pinned %s" % PINNED)
        return 0
    version = subprocess.run([protoc, "--version"], capture_output=True, text=True).stdout.strip()
    if version != PINNED:
        print("SKIP (explicit): protoc is %r but pin is %r; refusing to compare across versions" % (version, PINNED))
        return 0
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run([protoc, "-I", ".", "--python_out", tmp, PROTO_REL],
                       cwd=REPO_ROOT, check=True)
        fresh = os.path.join(tmp, OUT_REL)
        a = open(TRACKED, "rb").read()
        b = open(fresh, "rb").read()
        if a == b:
            print("PASS: et_def_pb2.py byte-identical to pinned-protoc repo-root regeneration (sha256 %s)" % hashlib.sha256(a).hexdigest())
            return 0
        print("FAIL: et_def_pb2.py drifted from pinned regeneration (tracked %s, fresh %s)"
              % (hashlib.sha256(a).hexdigest(), hashlib.sha256(b).hexdigest()))
        return 1

if __name__ == "__main__":
    sys.exit(main())
