#!/usr/bin/env python3
"""Replay the archived, uncorrected diagnostic in an isolated work directory."""
import argparse
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--compiler", default="g++")
    args = parser.parse_args()
    root = args.repo_root.resolve()
    build = args.build_dir.resolve()
    output = args.output_dir.resolve()
    output.relative_to(root / "work")
    output.mkdir(parents=True, exist_ok=True)
    archived = root / "die/docs/evidence/k-readiness-20261015/raw/probe.cc"
    source = archived.read_text()
    for suffix in ("die/tests/compiled_gemm_oracle_driver.cc", "die/src/GemmOSResourceFacts.hh"):
        source = source.replace("/workspace/template/" + suffix, str(root / suffix))
    prepared = output / "probe.cc"
    prepared.write_text(source)
    binary = output / "probe"
    compile_command = [args.compiler, "-std=c++17", "-O2", '-DCOARSE_DIE_SOURCE_DIR="' + str(root / "die") + '"', "-I", str(root / "die/include"), "-I", str(root / "astra-sim-wscllm/extern/helper"), str(prepared), str(build / "libcoarse_die.a"), "-o", str(binary)]
    with (output / "build.log").open("w") as log:
        subprocess.run(compile_command, cwd=root, stdout=log, stderr=subprocess.STDOUT, check=True)
    raw = output / "full_tile_k4096.json"
    with raw.open("w") as stream, (output / "full_tile_k4096.stderr").open("w") as error:
        subprocess.run([str(binary), "--m", "80", "--n", "128", "--k", "4096"], cwd=root, stdout=stream, stderr=error, check=True)
    subprocess.run([sys.executable, str(root / "die/validation/grouped_gemm_oracle.py"), "--raw", str(raw), "--output", str(output / "full_tile_k4096-report.json")], cwd=root, check=True)
    print("Diagnostic only; shared nonfinal-K output gate defect remains. See archived README.")


if __name__ == "__main__":
    main()
