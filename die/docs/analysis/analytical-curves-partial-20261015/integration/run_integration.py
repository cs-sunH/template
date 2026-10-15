#!/usr/bin/env python3
"""Plan or run portable five-repository ABI/config and FACE cost-path checks.

The default is dry-run. --execute runs only already-built libraries and source
tests; this runner does not configure/build, publish, synchronize, or edit source.
All generated stages, logs, maps and reports must live under repository work/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


VARIANTS = (
    "astra-sim-wscllm",
    "astra-sim-wscllm-LRU",
    "astra-sim-face",
    "astra-sim-face-LRU",
    "astra-sim-joint",
)
EXPECTED_CONFIG_TESTS = {
    "astra-sim-wscllm": 20,
    "astra-sim-wscllm-LRU": 15,
    "astra-sim-face": 18,
    "astra-sim-face-LRU": 17,
    "astra-sim-joint": 13,
}
FACE_MODULES = {
    "astra-sim-face": (
        "online.test_workspace_capacity_manager",
        "online.test_workspace_bridge",
        "online.test_die_model_adapter",
        "online.test_admission_liveness",
        "online.test_graph_batch_builder",
        "online.test_first_token_split",
        "online.test_train_machinery",
        "test_die_model_oracle",
        "test_face_scheduler",
        "test_face_kv_incremental_invariants",
        "online.test_future_workspace_manager",
        "online.test_manifest_workspace_envelope",
        "online.test_source_release_capacity",
    ),
    "astra-sim-face-LRU": (
        "online.test_workspace_capacity_manager",
        "online.test_workspace_bridge",
        "online.test_die_model_adapter",
        "online.test_admission_liveness",
        "online.test_graph_batch_builder",
        "online.test_first_token_split",
        "online.test_train_machinery",
        "test_die_model_oracle",
        "test_face_scheduler",
        "test_face_kv_incremental_invariants",
        "online.test_future_workspace_manager",
        "online.test_manifest_workspace_envelope",
        "online.test_source_release_capacity",
        "online.test_store_restore_ordering",
        "online.test_eviction_side_branch_structure",
        "online.test_transfer_frontier_neutral",
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def root_dir() -> Path:
    root = Path(__file__).resolve().parents[5]
    if not (root / "die/docs/analysis/runtime-credit-20261015/validate_five_abis.py").is_file():
        raise RuntimeError(f"cannot resolve repository root from this script: {root}")
    return root


def load_builds(path: Path, root: Path) -> dict[str, Path]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or set(raw) != set(VARIANTS):
        raise ValueError("build map must contain exactly the five active repositories")
    resolved = {}
    for variant in VARIANTS:
        candidate = Path(raw[variant])
        resolved[variant] = (candidate if candidate.is_absolute() else root / candidate).resolve()
    return resolved


def require_work_path(path: Path, root: Path) -> Path:
    resolved = path.resolve()
    work_root = (root / "work").resolve()
    if not resolved.is_relative_to(work_root) or resolved == work_root:
        raise ValueError(f"output directory must be a dedicated child of {work_root}")
    return resolved


def command_plan(root: Path, builds: dict[str, Path], old_coarse: Path | None,
                 output: Path) -> list[list[str]]:
    validator = root / "die/docs/analysis/runtime-credit-20261015/validate_five_abis.py"
    normalized_map = output / "build-map.absolute.json"
    commands: list[list[str]] = [[
        sys.executable, str(validator), "--repo-root", str(root),
        "--builds", str(normalized_map), "--output-dir", str(output / "five-abi"),
    ]]
    probe = Path(__file__).with_name("face_cost_abi_probe.py")
    for variant in ("astra-sim-face", "astra-sim-face-LRU"):
        coarse = builds[variant] / "lib/libcoarse_die_cabi.so"
        for expected, library in (("new", coarse), ("old", old_coarse)):
            if library is None:
                continue
            commands.append([
                sys.executable, str(probe), "--repo-root", str(root / variant),
                "--library", str(library), "--expect", expected,
                "--output", str(output / "face-cost" / f"{variant}-{expected}.json"),
            ])
    runner = root / "die/docs/analysis/face-workspace-20261015/run_python_checks.py"
    for variant in ("astra-sim-face", "astra-sim-face-LRU"):
        commands.append([
            sys.executable, str(runner), variant,
            "--repo-root", str(root), "--work-dir", str(output / "python" / variant),
            "--legacy-library", str(builds[variant] / "lib/libdie_model_cabi.so"),
            "--coarse-library", str(builds[variant] / "lib/libcoarse_die_cabi.so"),
            *FACE_MODULES[variant],
        ])
    return commands


def run(command: list[str], *, cwd: Path, env: dict[str, str], log: Path | None = None) -> str:
    result = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    output = result.stdout
    if log is not None:
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(output, encoding="utf-8")
    if result.returncode:
        raise RuntimeError(
            f"command failed ({result.returncode}): {' '.join(command)}\n{output}"
        )
    return output


def verify_config_results(path: Path) -> dict:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or {row.get("variant") for row in rows} != set(VARIANTS):
        raise AssertionError("five-ABI validation did not return all five active repositories")
    total = 0
    per_repo = {}
    for row in rows:
        variant = row["variant"]
        expected = EXPECTED_CONFIG_TESTS[variant]
        log_path = path.parent / f"{variant}-config-tests.log"
        log = log_path.read_text(encoding="utf-8")
        count = re.search(r"Ran (\d+) tests?", log)
        skipped = re.search(r"\(skipped=(\d+)\)", log)
        actual = int(count.group(1)) if count else None
        skip_count = int(skipped.group(1)) if skipped else 0
        if row.get("exit_code") != 0 or actual != expected or skip_count != 0:
            raise AssertionError(
                f"{variant}: exit={row.get('exit_code')} tests={actual}/{expected} skipped={skip_count}"
            )
        for abi_name in ("DIE_MODEL_CABI_PATH", "DIE_COARSE_CABI_PATH"):
            entry = row.get("libraries", {}).get(abi_name, {})
            if not entry.get("path") or not entry.get("sha256"):
                raise AssertionError(f"{variant}: missing actual-library identity for {abi_name}")
        per_repo[variant] = {"tests": actual, "skipped": skip_count}
        total += actual
    if total != 83:
        raise AssertionError(f"config resolver coverage changed: expected 83, got {total}")
    return {"tests": total, "skipped": 0, "per_repository": per_repo}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-map", type=Path, required=True,
                        help="JSON mapping the five exact repository names to existing build dirs")
    parser.add_argument("--old-coarse-library", type=Path,
                        help="real pre-cost-only coarse v2 library; required for --execute")
    parser.add_argument("--output-dir", type=Path,
                        help="new result directory under repository work/")
    parser.add_argument("--execute", action="store_true",
                        help="run the planned checks; without this flag only print the plan")
    args = parser.parse_args()

    root = root_dir()
    builds = load_builds(args.build_map.resolve(), root)
    default_output = (
        root / "work/analytical-curves/integration" /
        ("run-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    )
    output = require_work_path(args.output_dir or default_output, root)
    if args.execute and output.exists():
        raise SystemExit(f"refusing to reuse existing output directory: {output}")
    old_coarse = args.old_coarse_library.resolve() if args.old_coarse_library else None
    if args.execute and (old_coarse is None or not old_coarse.is_file()):
        raise SystemExit("--execute requires --old-coarse-library pointing at a real old coarse v2 .so")

    for variant, build in builds.items():
        for filename in ("lib/libdie_model_cabi.so", "lib/libcoarse_die_cabi.so"):
            path = build / filename
            if args.execute and not path.is_file():
                raise SystemExit(f"missing prebuilt ABI: {path}")
    if old_coarse and not old_coarse.is_file():
        raise SystemExit(f"old coarse library does not exist: {old_coarse}")

    plan = command_plan(root, builds, old_coarse, output)
    envelope = {
        "mode": "execute" if args.execute else "dry_run",
        "output_dir": str(output),
        "active_repositories": list(VARIANTS),
        "config_tests_expected": EXPECTED_CONFIG_TESTS,
        "config_tests_total_expected": 83,
        "old_coarse_library": str(old_coarse) if old_coarse else None,
        "commands": plan,
        "source_script_sha256": {
            str(Path(__file__).resolve()): sha256(Path(__file__).resolve()),
            str(Path(__file__).with_name("face_cost_abi_probe.py").resolve()):
                sha256(Path(__file__).with_name("face_cost_abi_probe.py").resolve()),
        },
    }
    if not args.execute:
        print(json.dumps(envelope, indent=2))
        return 0

    output.mkdir(parents=True, exist_ok=False)
    build_map_path = output / "build-map.absolute.json"
    build_map_path.write_text(json.dumps({k: str(v) for k, v in builds.items()},
                                         indent=2, sort_keys=True) + "\n",
                              encoding="utf-8")
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    # Existing validator writes its logs and structured results only under
    # this fresh work result directory.
    validator = root / "die/docs/analysis/runtime-credit-20261015/validate_five_abis.py"
    validator_command = [
        sys.executable, str(validator), "--repo-root", str(root),
        "--builds", str(build_map_path), "--output-dir", str(output / "five-abi"),
    ]
    run(validator_command, cwd=root, env=env,
        log=output / "five-abi-validator.stdout.log")
    config = verify_config_results(output / "five-abi/five-abi-config-results.json")

    probe = Path(__file__).with_name("face_cost_abi_probe.py").resolve()
    cost_results = []
    for variant in ("astra-sim-face", "astra-sim-face-LRU"):
        for expected, library in (
            ("new", builds[variant] / "lib/libcoarse_die_cabi.so"),
            ("old", old_coarse),
        ):
            assert library is not None
            target = output / "face-cost" / f"{variant}-{expected}.json"
            command = [
                sys.executable, str(probe), "--repo-root", str(root / variant),
                "--library", str(library), "--expect", expected,
                "--output", str(target),
            ]
            run(command, cwd=root, env=env,
                log=output / "logs" / f"{variant}-{expected}-cost-probe.log")
            cost_results.append(json.loads(target.read_text(encoding="utf-8")))

    python_results = []
    runner = root / "die/docs/analysis/face-workspace-20261015/run_python_checks.py"
    for variant in ("astra-sim-face", "astra-sim-face-LRU"):
        work_dir = output / "python" / variant
        command = [
            sys.executable, str(runner), variant,
            "--repo-root", str(root), "--work-dir", str(work_dir),
            "--legacy-library", str(builds[variant] / "lib/libdie_model_cabi.so"),
            "--coarse-library", str(builds[variant] / "lib/libcoarse_die_cabi.so"),
            *FACE_MODULES[variant],
        ]
        run(command, cwd=root, env=env,
            log=output / "logs" / f"{variant}-python.log")
        identity = json.loads((work_dir / "identity-and-command.json").read_text(encoding="utf-8"))
        test_output = output / "logs" / f"{variant}-python.log"
        test_log = test_output.read_text(encoding="utf-8")
        count = re.search(r"Ran (\d+) tests?", test_log)
        expected = 132 if variant == "astra-sim-face" else 151
        actual = int(count.group(1)) if count else None
        if actual != expected or re.search(r"\(skipped=(\d+)\)", test_log):
            raise AssertionError(f"{variant}: Python coverage {actual}/{expected} or skipped tests")
        python_results.append({
            "variant": variant,
            "tests": actual,
            "skipped": 0,
            "library_sha256": identity.get("library_sha256"),
            "identity_path": str(work_dir / "identity-and-command.json"),
        })

    envelope.update({
        "mode": "completed",
        "config_validation": config,
        "face_cost_probes": cost_results,
        "face_python": python_results,
        "limitations": [
            "This run covers focused ABI/config and FACE Python cost-path integration only.",
            "It does not certify full production workloads, target accuracy, host-budget limits, or all JOINT live-world imports.",
        ],
    })
    (output / "integration-results.json").write_text(
        json.dumps(envelope, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "output_dir": str(output),
        "config_tests": config["tests"],
        "face_cost_probes": len(cost_results),
        "face_python_tests": sum(row["tests"] for row in python_results),
        "result": str(output / "integration-results.json"),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
