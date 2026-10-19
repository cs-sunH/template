#!/bin/bash

set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "Usage: $0 <submodule-path> <patch-file>" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

SUBMODULE_PATH="${REPO_ROOT}/$1"
PATCH_FILE="${REPO_ROOT}/$2"

if [[ ! -e "${PATCH_FILE}" ]]; then
    echo "Patch file not found: ${PATCH_FILE}" >&2
    exit 1
fi

if [[ ! -d "${SUBMODULE_PATH}" ]]; then
    echo "Submodule path not found: ${SUBMODULE_PATH}" >&2
    exit 1
fi

cd "${SUBMODULE_PATH}"

if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "Refusing to apply patch series on a dirty worktree: ${SUBMODULE_PATH}" >&2
    echo "Commit, stash, or reset local changes first." >&2
    exit 1
fi

REBASE_APPLY_DIR="$(git rev-parse --git-path rebase-apply)"
REBASE_MERGE_DIR="$(git rev-parse --git-path rebase-merge)"

if [[ -d "${REBASE_APPLY_DIR}" ]] || [[ -d "${REBASE_MERGE_DIR}" ]]; then
    echo "An existing git am/rebase session is in progress in ${SUBMODULE_PATH}." >&2
    echo "Resolve it first, or run 'git am --abort' in the submodule." >&2
    exit 1
fi

git am --keep-cr --whitespace=nowarn --3way "${PATCH_FILE}"
