#!/usr/bin/env python3
"""Run LRU workspace-envelope tests against a caller-supplied public ABI.

The LRU checkout omits generated Chakra Python bindings. This runner compiles
its checked-in schema with protoc into LRU/work/chakra-python-root and copies
the exact LRU generate_trace.py source there. The copied module's normal root
discovery then resolves that real generated marker without repository edits.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import unittest


def find_repo_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if ((candidate / "die/src/Mapping.cc").is_file() and
                (candidate / "astra-sim-wscllm-LRU").is_dir()):
            return candidate
    raise RuntimeError(
        "cannot locate repository root; pass --repo-root explicitly")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root", type=Path,
        help="workspace root (defaults to discovery from this script path)")
    parser.add_argument(
        "--library", type=Path,
        help=("absolute public libcoarse_die_cabi.so; defaults to "
              "DIE_WORKSPACE_ENVELOPE_TEST_LIBRARY or DIE_COARSE_CABI_PATH"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    package_dir = Path(__file__).resolve().parent
    repo = (args.repo_root.resolve() if args.repo_root else
            find_repo_root(package_dir))
    lru = repo / "astra-sim-wscllm-LRU"
    source_root = lru / "sh_test_mesh/workload/llama2_7b_inference"
    online = source_root / "online"
    work_root = lru / "work/chakra-python-root"
    proto = lru / "extern/graph_frontend/chakra/schema/protobuf/et_def.proto"
    generator = source_root / "generate_trace.py"

    supplied_library = (args.library or
        (Path(os.environ["DIE_WORKSPACE_ENVELOPE_TEST_LIBRARY"])
         if os.environ.get("DIE_WORKSPACE_ENVELOPE_TEST_LIBRARY") else None) or
        (Path(os.environ["DIE_COARSE_CABI_PATH"])
         if os.environ.get("DIE_COARSE_CABI_PATH") else None))
    if supplied_library is None:
        raise SystemExit(
            "supply --library or an absolute "
            "DIE_WORKSPACE_ENVELOPE_TEST_LIBRARY/DIE_COARSE_CABI_PATH")
    if not supplied_library.is_absolute():
        raise SystemExit("the public ABI library path must be absolute")
    library = supplied_library.resolve(strict=True)
    if not library.is_file():
        raise SystemExit(f"public ABI library is not a file: {library}")
    os.environ["DIE_WORKSPACE_ENVELOPE_TEST_LIBRARY"] = str(library)
    os.environ["DIE_COARSE_CABI_PATH"] = str(library)

    protoc = shutil.which("protoc")
    if protoc is None:
        raise SystemExit("protoc is unavailable; activate the configured build environment")
    if not proto.is_file() or not generator.is_file():
        raise SystemExit("the LRU checkout is missing its Chakra schema or generator source")
    work_root.mkdir(parents=True, exist_ok=True)
    relative_proto = proto.relative_to(lru)
    subprocess.run([
        protoc, f"--proto_path={lru}", f"--python_out={work_root}",
        str(relative_proto),
    ], check=True, cwd=lru)

    # Keep the source bytes exact while giving generate_trace.py a legitimate
    # project root that contains the generated protobuf marker.
    work_generator = work_root / "generate_trace.py"
    shutil.copy2(generator, work_generator)
    if work_generator.read_bytes() != generator.read_bytes():
        raise RuntimeError("generated test setup has a modified generator copy")

    sys.path.insert(0, str(work_root))
    sys.path.insert(0, str(online))
    import generate_trace  # noqa: E402

    expected_module = work_root / relative_proto.with_name("et_def_pb2.py")
    module_name = "extern.graph_frontend.chakra.schema.protobuf.et_def_pb2"
    actual_module = Path(sys.modules[module_name].__file__).resolve()
    if actual_module != expected_module.resolve():
        raise RuntimeError(
            f"Chakra module came from {actual_module}, expected {expected_module}")
    if generate_trace.PROJECT_ROOT.resolve() != work_root.resolve():
        raise RuntimeError(
            f"generator resolved unexpected root: {generate_trace.PROJECT_ROOT}")

    print(f"public ABI library: {library}", flush=True)
    print(f"protoc schema output: {actual_module}", flush=True)
    print(f"generator source copy: {work_generator}", flush=True)
    print(f"generator project root: {generate_trace.PROJECT_ROOT}", flush=True)

    os.chdir(repo)
    unittest.main(
        module=None,
        argv=[sys.argv[0], "test_die_workspace_envelope",
              "test_manifest_workspace_envelope"],
        verbosity=2,
    )
    return 0


if __name__ == "__main__":
    main()
