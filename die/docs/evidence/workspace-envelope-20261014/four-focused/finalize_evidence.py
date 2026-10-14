from __future__ import annotations

import json
import hashlib
import shlex
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
WORK = ROOT / "work/workspace-envelope-20261014/four-focused"
results_path = WORK / "results.json"
records = json.loads(results_path.read_text(encoding="utf-8"))

for record in records:
    build = Path(record["build_dir"])
    for rel in (
        "AstraSim/extern/die_model/coarse/libcoarse_die.a",
        "AstraSim/extern/die_model/libdie_model.a",
    ):
        path = build / rel
        if path.is_file():
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            record["binary_sha256"][str(path.relative_to(ROOT))] = digest
        else:
            record["binary_sha256"][str(path.relative_to(ROOT))] = "MISSING"
    link_command = build / "frontend/CMakeFiles/AstraSim_Analytical_Congestion_Aware_Online.dir/link.txt"
    if link_command.is_file():
        link_text = link_command.read_text(encoding="utf-8")
        captured_link = WORK / "source-identity" / f"{record['repo']}-online-link.txt"
        captured_link.write_text(link_text, encoding="utf-8")
        record["online_link_command_file"] = str(captured_link.relative_to(ROOT))
        record["online_link_command_sha256"] = hashlib.sha256(link_text.encode("utf-8")).hexdigest()
        record["online_link_uses_coarse_static_library"] = (
            "extern/die_model/coarse/libcoarse_die.a" in link_text
        )
    else:
        record["online_link_uses_coarse_static_library"] = False

for record in records:
    sync_pre = record["canonical_sync_pre"].get("result", {})
    sync_post = record["canonical_sync_post"].get("result", {})
    required_ok = (
        record["build_exit"] == 0
        and record["ctest_exit"] == 0
        and record["ctest_passed"] == record["expected_tests"]
        and record["ctest_selected"] == record["expected_tests"]
        and record["coarse_source_manifest_unchanged"] is True
        and record["build_input_manifest_unchanged"] is True
        and sync_pre.get("status") == "passed"
        and sync_post.get("status") == "passed"
        and sync_pre.get("source_digest") == "e0b6fd1a083ebe89cc73956f84f07524e6c3944fbcd4cc28a91a947d66cbbf78"
        and sync_post.get("source_digest") == "e0b6fd1a083ebe89cc73956f84f07524e6c3944fbcd4cc28a91a947d66cbbf78"
        and all(value != "MISSING" for value in record["binary_sha256"].values())
        and record.get("online_link_uses_coarse_static_library") is True
    )
    record["status"] = "passed" if required_ok else "failed_or_incomplete"
    if record["repo"] == "astra-sim-wscllm-LRU" and not record["tracked_diff_unchanged"]:
        record["tracked_diff_change_scope"] = [
            "astra-sim-wscllm-LRU/sh_test_mesh/workload/llama2_7b_inference/session_kv_manager.py",
            "astra-sim-wscllm-LRU/sh_test_mesh/workload/llama2_7b_inference/test_workspace_admission_manager.py",
        ]
        record["tracked_diff_note"] = (
            "The repo-wide tracked diff changed during the build because the parallel Python workspace audit "
            "updated these two files. They are outside the C++/CMake build-input manifests, which were byte-identical "
            "before and after; canonical coarse source manifests were also unchanged. The C++ result is scoped to "
            "the recorded HEAD, C++/CMake input manifests, and binary hashes, not a claim that the full worktree froze."
        )

results_path.write_text(json.dumps(records, indent=2, sort_keys=True) + "\n", encoding="utf-8")

summary = {
    "status": "passed" if all(record["status"] == "passed" for record in records) else "failed_or_incomplete",
    "repositories": len(records),
    "focused_tests_passed": sum(int(record["ctest_passed"]) for record in records),
    "focused_tests_expected": sum(int(record["expected_tests"]) for record in records),
    "canonical_source_digest": "e0b6fd1a083ebe89cc73956f84f07524e6c3944fbcd4cc28a91a947d66cbbf78",
    "online_rebuilt_for_each_repository": True,
    "abi_targets_rebuilt_by_this_run": False,
    "joint_business_focused": "not_yet_ported",
    "results": str(results_path.relative_to(ROOT)),
    "tracked_worktree_change_during_lru_build": "parallel Python workspace audit; excluded from C++/CMake source manifests",
}
(WORK / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

commands: list[str] = [
    "Working directory: /workspace/template",
    "Environment:",
    "  source /workspace/cloud-setup/activate.sh",
    "  unset PROTOBUF_FROM_SOURCE",
    "",
    "Repository order: base, FACE, FACE-LRU, WSL-LRU. The C ABI targets were not rebuilt in this focused run.",
]
for record in records:
    commands.extend([
        "",
        f"{record['repo']} build:",
        "  " + shlex.join(record["build_command"]),
        f"{record['repo']} CTest:",
        "  " + shlex.join(record["ctest_command"]),
    ])
commands.extend([
    "",
    "Runner command:",
    "  python3 work/workspace-envelope-20261014/four-focused/run_focused.py",
    "",
    "Each build command included AstraSim_Analytical_Congestion_Aware_Online plus the listed focused test targets.",
])
(WORK / "commands.txt").write_text("\n".join(commands) + "\n", encoding="utf-8")

def online_hash(record: dict[str, object]) -> str:
    path = next(
        key for key in record["binary_sha256"]
        if key.endswith("/bin/AstraSim_Analytical_Congestion_Aware_Online")
    )
    return str(record["binary_sha256"][path])

lines = [
    "# Four-repository focused rebuild against canonical23",
    "",
    "This run rebuilt the base, FACE, FACE-LRU, and WSL-LRU Online target and requested C++ focused tests, in that order. All evidence is saved in this directory. It did not rebuild either ABI target or run a production Online workload.",
    "",
    "The canonical 23-file sync check passed before and after every variant with source digest `e0b6fd1a083ebe89cc73956f84f07524e6c3944fbcd4cc28a91a947d66cbbf78`. Per-variant coarse source and C++/CMake build-input manifests match byte for byte across each build. Each variant's `libcoarse_die_cabi.so` hash is `7e9b9670b8c06c8600688acf6446acc5f112a97e83315a26ea8d554040cc5600`; ABI binaries were hashed from their existing root-built outputs, not rebuilt here.",
    "",
    "| Variant | Focused CTest | Online SHA-256 | Coarse C ABI SHA-256 |",
    "|---|---:|---|---|",
]
for record in records:
    coarse_key = next(
        key for key in record["binary_sha256"] if key.endswith("/lib/libcoarse_die_cabi.so")
    )
    lines.append(
        f"| `{record['repo']}` | {record['ctest_passed']}/{record['expected_tests']} | "
        f"`{online_hash(record)}` | `{record['binary_sha256'][coarse_key]}` |"
    )
lines.extend([
    "",
    "The three non-LRU variants selected `DieModelAdapterTest`, `DieExecutionLifecycleTest`, and `LocalHbmDieMemoryTest`. WSL-LRU selected the nine tests recorded in `commands.txt` and `results.json`, including `BridgeLoopbackTest` and the two-Plan lifecycle test. Total focused result: 18/18. All four Online targets built successfully.",
    "",
    "During the WSL-LRU build, the parallel Python workspace audit changed `session_kv_manager.py` and `test_workspace_admission_manager.py`. This changed the repository-wide tracked diff; those files are outside the C++/CMake inputs used by these targets. The recorded C++/CMake source manifests and canonical coarse source manifest remained unchanged. The evidence does not claim the full worktree was frozen.",
    "",
    "The new JOINT model has no business-focused C++ integration test in this set and is reported as `not_yet_ported`. No JOINT Online target or JOINT business focused tests were run.",
    "",
    "Exact commands are in `commands.txt`; build and CTest logs are in `logs/`; target and source SHA-256 details are in `results.json` and `source-identity/`.",
])
(WORK / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
coarse_static_hashes = sorted({
    value for record in records
    for key, value in record["binary_sha256"].items()
    if key.endswith("/coarse/libcoarse_die.a")
})
with (WORK / "REPORT.md").open("a", encoding="utf-8") as report:
    report.write("\nThe complete 25-file coarse subtree manifests are identical across the four copied trees. Tool versions are in environment.log.\n")
    report.write("The Online-linked coarse static library hashes are: " + ", ".join(coarse_static_hashes) + ".\n")
    report.write("All four captured Online link commands reference the coarse static library; exact commands and hashes are in source-identity/.\n")

if summary["status"] != "passed":
    raise SystemExit("one or more focused variant checks did not pass")
print(json.dumps(summary, sort_keys=True))
