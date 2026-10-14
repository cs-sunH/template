from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
IDENTITY = ROOT / "work/workspace-envelope-20261014/four-focused/source-identity"
REPOS = [
    "astra-sim-wscllm",
    "astra-sim-face",
    "astra-sim-face-LRU",
    "astra-sim-wscllm-LRU",
]


def read_manifest(repo: str) -> dict[str, str]:
    path = IDENTITY / f"{repo}-pre-coarse.sha256"
    result: dict[str, str] = {}
    prefix = repo + "/"
    for line in path.read_text(encoding="utf-8").splitlines():
        digest, file_path = line.split("  ", 1)
        if not file_path.startswith(prefix):
            raise ValueError(f"unexpected manifest path: {file_path}")
        result[file_path[len(prefix):]] = digest
    return result


manifests = {repo: read_manifest(repo) for repo in REPOS}
reference_repo = REPOS[0]
reference = manifests[reference_repo]
differences = {
    repo: sorted(
        key for key in set(reference) | set(manifests[repo])
        if reference.get(key) != manifests[repo].get(key)
    )
    for repo in REPOS[1:]
}
result = {
    "reference_repo": reference_repo,
    "files_per_variant": {repo: len(manifest) for repo, manifest in manifests.items()},
    "identical_to_reference": {repo: not paths for repo, paths in differences.items()},
    "differences": differences,
    "all_identical": all(not paths for paths in differences.values())
        and all(len(manifest) == len(reference) for manifest in manifests.values()),
}
output = IDENTITY / "coarse-manifest-comparison.json"
output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(result, sort_keys=True))
if not result["all_identical"]:
    raise SystemExit(1)
