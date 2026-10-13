#!/usr/bin/env python3
"""Publish one content-verified coarse source tree into all five repositories.

Legacy die_model sources remain available for explicit legacy profiles. Every
coarse source comes from template/die, and each repository can build offline.
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

REPOSITORIES = ("astra-sim-wscllm", "astra-sim-wscllm-LRU", "astra-sim-face",
                "astra-sim-face-LRU", "astra-sim-joint")
SOURCE_DIRS = ("include", "src", "configs")

def release_files(source):
    files = [source / "CMakeLists.txt"]
    for name in SOURCE_DIRS:
        files += sorted(p for p in (source / name).rglob("*") if p.is_file())
    return files

def digest(files, source):
    entries = {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    identity = hashlib.sha256(json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {"schema": "coarse_die_release_v2", "source_digest": identity, "files": entries}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[1]
    root = source.parent
    manifest = digest(release_files(source), source)
    for repo in REPOSITORIES:
        parent = root / repo / "extern/die_model"
        target = parent / "coarse"
        if not args.check:
            target.mkdir(parents=True, exist_ok=True)
            # Remove obsolete canonical release files, never arbitrary adapter files.
            old = target / "RELEASE.json"
            if old.exists():
                for rel in json.loads(old.read_text()).get("files", {}):
                    if rel not in manifest["files"]:
                        (target / rel).unlink(missing_ok=True)
            for rel in manifest["files"]:
                dest = target / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source / rel, dest)
            (target / "RELEASE.json").write_text(json.dumps(manifest, indent=2)+"\n")
            cmake = parent / "CMakeLists.txt"
            text = cmake.read_text()
            marker = '# Canonical coarse_die_v2 release (synced from template/die).'
            if marker not in text:
                cmake.write_text(text+"\n"+marker+"\nset(COARSE_DIE_BUILD_TESTS OFF CACHE BOOL \"Standalone tests live in template/die\")\nadd_subdirectory(coarse)\n")
        for rel, expected in manifest["files"].items():
            p = target / rel
            if not p.exists() or hashlib.sha256(p.read_bytes()).hexdigest() != expected:
                raise SystemExit(f"source mismatch: {repo}/extern/die_model/coarse/{rel}")
        installed = json.loads((target / "RELEASE.json").read_text())
        if installed != manifest:
            raise SystemExit(f"release identity mismatch: {repo}")
    print(json.dumps({"status": "passed", "repositories": REPOSITORIES,
                      "source_digest": manifest["source_digest"], "files": len(manifest["files"])}))

if __name__ == "__main__":
    main()
