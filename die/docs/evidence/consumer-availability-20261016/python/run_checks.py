#!/usr/bin/env python3
"""Run exact staged Python consumers; no SDK build or performance timing."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("red", "green"))
    parser.add_argument("--scope", choices=("regression", "all"), default="regression")
    parser.add_argument("--legacy-library", type=Path)
    parser.add_argument("--coarse-library", type=Path)
    parser.add_argument("--run-label")
    args = parser.parse_args()
    evidence = Path(__file__).resolve().parent
    root = evidence.parents[4]
    run_name = args.run_label or args.phase + "-" + args.scope
    if not run_name or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-" for c in run_name):
        raise SystemExit("run-label must use only lowercase letters, digits and hyphens")
    work = root / "work/consumer-availability/python" / run_name
    logs = evidence / run_name
    if work.exists() or logs.exists():
        raise SystemExit("preserve previous evidence: choose a new run location")
    logs.mkdir(parents=True)
    results = []
    for variant in ("astra-sim-face", "astra-sim-face-LRU", "astra-sim-joint"):
        source = root / variant
        stage = work / variant / "stage"
        relative_workload = Path("sh_test_mesh/workload/llama2_7b_inference")
        hashes = {}
        for path in sorted((source / "sh_test_mesh").rglob("*")):
            if not path.is_file() or path.suffix not in {
                    ".py", ".csv", ".json", ".yml", ".yaml"}:
                continue
            relative = path.relative_to(source)
            if any(part in {"evidence", "runs", "__pycache__", "generated"}
                   for part in relative.parts):
                continue
            target = stage / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            hashes[str(relative)] = sha(path)
            if target.read_bytes() != path.read_bytes():
                raise RuntimeError("staged source mismatch: " + str(relative))
        proto = Path("extern/graph_frontend/chakra/schema/protobuf/et_def.proto")
        proto_command = ["protoc", "--proto_path=" + str(source),
                         "--python_out=" + str(stage), str(proto)]
        subprocess.run(proto_command, cwd=source, check=True)
        shutil.copytree(source / "extern/die_model/coarse/configs",
                        stage / "extern/die_model/coarse/configs")
        environment = os.environ.copy()
        environment["PYTHONPATH"] = os.pathsep.join(map(str, (
            stage, stage / "sh_test_mesh", stage / relative_workload)))
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        libraries = {}
        for name, path in (("DIE_MODEL_CABI_PATH", args.legacy_library),
                           ("DIE_COARSE_CABI_PATH", args.coarse_library)):
            if path is not None:
                path = path.resolve(strict=True)
                environment[name] = str(path)
                libraries[name] = {"path": str(path), "sha256": sha(path)}
            else:
                environment.pop(name, None)
        if variant == "astra-sim-joint":
            modules = ["online.test_coarse_die_adapter" + (
                ".CoarseOracleTests" if args.scope == "regression" else "")]
        else:
            modules = (["test_die_model_oracle.CoarseAvailabilityTests"]
                       if args.scope == "regression" else
                       ["test_die_model_oracle", "test_face_scheduler",
                        "online.test_die_model_adapter"])
        command = [sys.executable, "-m", "unittest", "-v", *modules]
        prefix = logs / variant
        with prefix.with_suffix(".stdout.log").open("w") as stdout, \
                prefix.with_suffix(".stderr.log").open("w") as stderr:
            completed = subprocess.run(command, cwd=stage / relative_workload,
                                       env=environment, stdout=stdout, stderr=stderr)
        identity = {
            "variant": variant, "phase": args.phase, "scope": args.scope,
            "source_base_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
            "source_sha256": hashes, "proto_sha256": sha(source / proto),
            "protobuf_generation_command": proto_command,
            "library_identity": libraries,
            "native_library_scope": ("mocked batch responses only; no SDK loaded"
                                     if args.scope == "regression" else
                                     "real libraries for existing FACE tests; new status regressions mock batch responses"),
            "command": command, "cwd": str(stage / relative_workload),
            "environment": {key: environment[key] for key in (
                "PYTHONPATH", "PYTHONDONTWRITEBYTECODE", "DIE_MODEL_CABI_PATH",
                "DIE_COARSE_CABI_PATH") if key in environment},
            "exit_code": completed.returncode,
        }
        prefix.with_suffix(".json").write_text(json.dumps(identity, indent=2) + "\n")
        for relative in hashes:
            if Path(relative).name in {"die_model_oracle.py", "test_die_model_oracle.py",
                                       "test_coarse_die_adapter.py"}:
                saved = logs / "source" / variant / relative
                saved.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(stage / relative, saved)
        results.append({"variant": variant, "exit_code": completed.returncode})
    (logs / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results))
    expected_failure = args.phase == "red"
    return 0 if all((row["exit_code"] != 0) == expected_failure for row in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
