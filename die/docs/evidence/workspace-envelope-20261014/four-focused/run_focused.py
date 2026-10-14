from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
WORK = ROOT / "work/workspace-envelope-20261014/four-focused"
CANONICAL_DIGEST = "e0b6fd1a083ebe89cc73956f84f07524e6c3944fbcd4cc28a91a947d66cbbf78"
VARIANTS = [
    {
        "repo": "astra-sim-wscllm",
        "build": ROOT / "work/online-baseline-20261014/integration-build/build/astra-sim-wscllm",
        "targets": [
            "AstraSim_Analytical_Congestion_Aware_DieModelAdapterTest",
            "AstraSim_Analytical_Congestion_Aware_DieExecutionLifecycleTest",
            "AstraSim_Analytical_Congestion_Aware_LocalHbmDieMemoryTest",
        ],
        "pattern": "(DieModelAdapter|DieExecutionLifecycle|LocalHbmDieMemory)Test",
        "expected": 3,
    },
    {
        "repo": "astra-sim-face",
        "build": ROOT / "work/five-os-20261014/build/astra-sim-face",
        "targets": [
            "AstraSim_Analytical_Congestion_Aware_DieModelAdapterTest",
            "AstraSim_Analytical_Congestion_Aware_DieExecutionLifecycleTest",
            "AstraSim_Analytical_Congestion_Aware_LocalHbmDieMemoryTest",
        ],
        "pattern": "(DieModelAdapter|DieExecutionLifecycle|LocalHbmDieMemory)Test",
        "expected": 3,
    },
    {
        "repo": "astra-sim-face-LRU",
        "build": ROOT / "work/five-os-20261014/build/astra-sim-face-LRU",
        "targets": [
            "AstraSim_Analytical_Congestion_Aware_DieModelAdapterTest",
            "AstraSim_Analytical_Congestion_Aware_DieExecutionLifecycleTest",
            "AstraSim_Analytical_Congestion_Aware_LocalHbmDieMemoryTest",
        ],
        "pattern": "(DieModelAdapter|DieExecutionLifecycle|LocalHbmDieMemory)Test",
        "expected": 3,
    },
    {
        "repo": "astra-sim-wscllm-LRU",
        "build": ROOT / "work/lru-cpp-liveness-pressure-20261014/two-plan-confirmation/build/aggregate",
        "targets": [
            "AstraSim_Analytical_WorkspaceAdmissionTest",
            "AstraSim_Analytical_Congestion_Aware_DecisionMailboxTest",
            "AstraSim_Analytical_Congestion_Aware_GraphBatchCommitterTest",
            "AstraSim_Analytical_Congestion_Aware_ParsedBatchTest",
            "AstraSim_Analytical_Congestion_Aware_LocalHbmBandwidthModelTest",
            "AstraSim_Analytical_Congestion_Aware_LocalHbmModelTest",
            "AstraSim_Analytical_Congestion_Aware_DieModelAdapterTest",
            "AstraSim_Analytical_Congestion_Aware_DieExecutionLifecycleTest",
            "AstraSim_Analytical_Congestion_Aware_BridgeLoopbackTest",
        ],
        "pattern": "(WorkspaceAdmissionTest|DecisionMailboxTest|GraphBatchCommitterTest|ParsedBatchTest|LocalHbmBandwidthModelTest|LocalHbmModelTest|DieModelAdapterTest|DieExecutionLifecycleTest|BridgeLoopbackTest)",
        "expected": 9,
    },
]

ONLINE_TARGET = "AstraSim_Analytical_Congestion_Aware_Online"
ENV = dict(os.environ)
ENV.pop("PROTOBUF_FROM_SOURCE", None)
LOGS = WORK / "logs"
IDENTITY = WORK / "source-identity"
LOGS.mkdir(parents=True, exist_ok=True)
IDENTITY.mkdir(parents=True, exist_ok=True)


def run_logged(command: list[str], path: Path) -> subprocess.CompletedProcess[str]:
    with path.open("w", encoding="utf-8") as stream:
        return subprocess.run(
            command,
            cwd=ROOT,
            env=ENV,
            stdout=stream,
            stderr=subprocess.STDOUT,
            text=True,
        )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def snapshot_coarse(repo: str, label: str) -> Path:
    root = ROOT / repo
    coarse = root / "extern/die_model/coarse"
    paths = [root / "extern/die_model/CMakeLists.txt"]
    paths.extend(path for path in coarse.rglob("*") if path.is_file())
    paths = sorted(paths)
    output = IDENTITY / f"{repo}-{label}-coarse.sha256"
    with output.open("w", encoding="utf-8") as stream:
        for path in paths:
            stream.write(f"{sha256(path)}  {path.relative_to(ROOT)}\n")
    return output


def snapshot_integration_sources(repo: str, label: str) -> Path:
    root = ROOT / repo
    candidates: set[Path] = set()
    for base in (
        root / "CMakeLists.txt",
        root / "astra-sim",
        root / "extern/die_model/coarse",
        root / "extern/die_model/CMakeLists.txt",
    ):
        if base.is_file():
            candidates.add(base)
        elif base.is_dir():
            for path in base.rglob("*"):
                if not path.is_file():
                    continue
                if path.suffix.lower() in {".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".hxx"} or path.name == "CMakeLists.txt":
                    candidates.add(path)
    output = IDENTITY / f"{repo}-{label}-build-inputs.sha256"
    with output.open("w", encoding="utf-8") as stream:
        for path in sorted(candidates):
            stream.write(f"{sha256(path)}  {path.relative_to(ROOT)}\n")
    return output


def capture_git_identity(repo: str, label: str) -> dict[str, str]:
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip()
    status = subprocess.check_output(["git", "status", "--short", "--", repo], cwd=ROOT, text=True)
    diff = subprocess.check_output(["git", "diff", "--binary", "--", repo], cwd=ROOT)
    patch_path = IDENTITY / f"{repo}-{label}-tracked-diff.patch"
    patch_path.write_bytes(diff)
    status_path = IDENTITY / f"{repo}-{label}-status.txt"
    status_path.write_text(status, encoding="utf-8")
    return {
        "head": head,
        "branch": branch,
        "status_file": str(status_path.relative_to(ROOT)),
        "tracked_diff_file": str(patch_path.relative_to(ROOT)),
        "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
    }


def check_canonical(repo: str, label: str) -> dict[str, object]:
    path = LOGS / f"{repo}-canonical-sync-{label}.log"
    result = run_logged(["python3", "die/tools/sync_repositories.py", "--check"], path)
    text = path.read_text(encoding="utf-8", errors="replace")
    try:
        payload = json.loads(text.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        payload = {"raw_output": text[-2000:]}
    return {
        "exit": result.returncode,
        "log": str(path.relative_to(ROOT)),
        "result": payload,
    }


def binary_hashes(build: Path, targets: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for target in targets:
        path = build / "bin" / target
        if path.is_file():
            result[str(path.relative_to(ROOT))] = sha256(path)
        else:
            result[str(path.relative_to(ROOT))] = "MISSING"
    for rel in (
        "lib/libcoarse_die_cabi.so",
        "lib/libdie_model_cabi.so",
        "AstraSim/extern/die_model/coarse/libcoarse_die.a",
        "AstraSim/extern/die_model/libdie_model.a",
    ):
        path = build / rel
        if path.is_file():
            result[str(path.relative_to(ROOT))] = sha256(path)
        else:
            result[str(path.relative_to(ROOT))] = "MISSING"
    return result


records: list[dict[str, object]] = []
result_path = WORK / "results.json"


def save_results() -> None:
    result_path.write_text(json.dumps(records, indent=2, sort_keys=True) + "\n", encoding="utf-8")


for variant in VARIANTS:
    repo = str(variant["repo"])
    build = Path(variant["build"])
    test_targets = list(variant["targets"])
    expected = int(variant["expected"])
    repo_dir = ROOT / repo
    coarse_pre = snapshot_coarse(repo, "pre")
    source_pre = snapshot_integration_sources(repo, "pre")
    git_pre = capture_git_identity(repo, "pre")
    sync_pre = check_canonical(repo, "pre")
    sync_payload = sync_pre["result"]
    source_digest = sync_payload.get("source_digest") if isinstance(sync_payload, dict) else None
    if sync_pre["exit"] != 0 or source_digest != CANONICAL_DIGEST:
        raise RuntimeError(f"{repo}: canonical source check failed; see {sync_pre['log']}: {sync_payload}")

    targets = [*test_targets, ONLINE_TARGET]
    build_command = ["cmake", "--build", str(build), "--parallel", "2", "--target", *targets]
    build_log = LOGS / f"{repo}-build.log"
    built = run_logged(build_command, build_log)
    record: dict[str, object] = {
        "repo": repo,
        "build_dir": str(build),
        "repo_head": git_pre["head"],
        "branch": git_pre["branch"],
        "canonical_source_digest": source_digest,
        "canonical_sync_pre": sync_pre,
        "coarse_source_manifest_pre": str(coarse_pre.relative_to(ROOT)),
        "build_input_manifest_pre": str(source_pre.relative_to(ROOT)),
        "git_identity_pre": git_pre,
        "test_targets": test_targets,
        "online_target": ONLINE_TARGET,
        "build_command": build_command,
        "build_exit": built.returncode,
        "build_log": str(build_log.relative_to(ROOT)),
        "expected_tests": expected,
    }
    if built.returncode != 0:
        record["status"] = "build_failed"
        record["binary_sha256"] = binary_hashes(build, targets)
        records.append(record)
        save_results()
        print(json.dumps(record, sort_keys=True), flush=True)
        sys.exit(built.returncode)

    test_command = [
        "ctest", "--test-dir", str(build), "--output-on-failure", "-R", str(variant["pattern"])
    ]
    test_log = LOGS / f"{repo}-ctest.log"
    tested = run_logged(test_command, test_log)
    text = test_log.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"100% tests passed, (\d+) tests? failed out of (\d+)", text)
    if match:
        failed, total = map(int, match.groups())
        passed = total - failed
    else:
        total = len(re.findall(r"\d+/\d+ Test #\d+:", text))
        passed = len(re.findall(r"Test #\d+: .* Passed", text))
    record.update({
        "ctest_command": test_command,
        "ctest_exit": tested.returncode,
        "ctest_log": str(test_log.relative_to(ROOT)),
        "ctest_passed": passed,
        "ctest_selected": total,
        "binary_sha256": binary_hashes(build, targets),
    })

    coarse_post = snapshot_coarse(repo, "post")
    source_post = snapshot_integration_sources(repo, "post")
    git_post = capture_git_identity(repo, "post")
    sync_post = check_canonical(repo, "post")
    record.update({
        "coarse_source_manifest_post": str(coarse_post.relative_to(ROOT)),
        "build_input_manifest_post": str(source_post.relative_to(ROOT)),
        "coarse_source_manifest_unchanged": coarse_pre.read_bytes() == coarse_post.read_bytes(),
        "build_input_manifest_unchanged": source_pre.read_bytes() == source_post.read_bytes(),
        "git_identity_post": git_post,
        "tracked_diff_unchanged": git_pre["tracked_diff_sha256"] == git_post["tracked_diff_sha256"],
        "canonical_sync_post": sync_post,
    })
    record["status"] = "passed" if (
        tested.returncode == 0
        and passed == expected
        and total == expected
        and record["coarse_source_manifest_unchanged"]
        and record["build_input_manifest_unchanged"]
        and sync_post["exit"] == 0
        and isinstance(sync_post["result"], dict)
        and sync_post["result"].get("source_digest") == CANONICAL_DIGEST
    ) else "failed_or_incomplete"
    if not record["tracked_diff_unchanged"]:
        record["tracked_diff_note"] = (
            "The repo-wide tracked diff changed during this run, while the complete C++/CMake build-input "
            "manifest and canonical coarse source manifest were byte-identical before and after. The known "
            "concurrent LRU changes were Python session_kv_manager/test sources, outside these C++ targets."
            if repo == "astra-sim-wscllm-LRU"
            else "The repo-wide tracked diff changed during the run; review the captured build-input manifests."
        )
    records.append(record)
    save_results()
    print(json.dumps(record, sort_keys=True), flush=True)
    if record["status"] != "passed":
        sys.exit(1)

summary = {
    "status": "passed",
    "repositories": len(records),
    "focused_tests_passed": sum(int(r["ctest_passed"]) for r in records),
    "focused_tests_expected": sum(int(r["expected_tests"]) for r in records),
    "canonical_source_digest": CANONICAL_DIGEST,
    "online_rebuilt_for_each_repository": True,
    "joint_business_focused": "not_yet_ported",
    "results": str(result_path.relative_to(ROOT)),
}
summary_path = WORK / "summary.json"
summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(summary, sort_keys=True), flush=True)
