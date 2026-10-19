#!/bin/bash

set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: $0 <submodule-path> <patch-file> [fallback-remote-ref]" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SUBMODULE_REL="$1"
PATCH_FILE="${REPO_ROOT}/$2"
FALLBACK_REMOTE_REF="${3:-}"
SUBMODULE_PATH="${REPO_ROOT}/${SUBMODULE_REL}"

if [[ ! -d "${SUBMODULE_PATH}/.git" && ! -f "${SUBMODULE_PATH}/.git" ]]; then
  echo "Submodule path is not a git worktree: ${SUBMODULE_PATH}" >&2
  exit 1
fi

cd "${SUBMODULE_PATH}"

resolve_base_commit() {
  local gitlink_base=""
  local merge_base=""

  gitlink_base="$(git -C "${REPO_ROOT}" ls-tree HEAD -- "${SUBMODULE_REL}" | awk '{print $3}')"
  if [[ -n "${gitlink_base}" ]] \
    && git cat-file -e "${gitlink_base}^{commit}" 2>/dev/null \
    && git merge-base --is-ancestor "${gitlink_base}" HEAD
  then
    echo "${gitlink_base}"
    return 0
  fi

  if [[ -n "${FALLBACK_REMOTE_REF}" ]] \
    && git rev-parse --verify "${FALLBACK_REMOTE_REF}^{commit}" >/dev/null 2>&1
  then
    merge_base="$(git merge-base HEAD "${FALLBACK_REMOTE_REF}" || true)"
    if [[ -n "${merge_base}" ]]; then
      echo "${merge_base}"
      return 0
    fi
  fi

  return 1
}

BASE="$(resolve_base_commit || true)"
if [[ -z "${BASE}" ]]; then
  echo "Failed to determine patch base for ${SUBMODULE_REL}." >&2
  echo "Tried superproject gitlink and fallback remote ref '${FALLBACK_REMOTE_REF}'." >&2
  exit 1
fi

if [[ "${BASE}" == "$(git rev-parse HEAD)" ]]; then
  echo "No commits to export for ${SUBMODULE_REL}; HEAD matches base ${BASE}." >&2
  exit 1
fi

PATCH_DIR="$(dirname "${PATCH_FILE}")"
mkdir -p "${PATCH_DIR}"
TMP_PATCH_FILE="$(mktemp "${PATCH_DIR}/.$(basename "${PATCH_FILE}").tmp.XXXXXX")"
trap 'rm -f "${TMP_PATCH_FILE}"' EXIT

git format-patch --base="${BASE}" --stdout "${BASE}"..HEAD > "${TMP_PATCH_FILE}"
mv "${TMP_PATCH_FILE}" "${PATCH_FILE}"
chmod 644 "${PATCH_FILE}"
trap - EXIT

echo "Exported ${SUBMODULE_REL} patch series against base ${BASE} to ${PATCH_FILE}."
