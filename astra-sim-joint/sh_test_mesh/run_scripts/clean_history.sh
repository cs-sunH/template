#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
SH_TEST_DIR=$(cd -- "${SCRIPT_DIR}/.." && pwd -P)

DRY_RUN=0

usage() {
  cat <<'EOF'
Usage: clean_history.sh [--dry-run]

Remove all historical contents under sh_test_mesh/generated and
sh_test_mesh/results while preserving the two directories themselves.

Options:
  --dry-run  Show the entries that would be removed without deleting them.
  -h, --help Show this help message.
EOF
}

case "${1:-}" in
  "")
    ;;
  --dry-run)
    DRY_RUN=1
    ;;
  -h|--help)
    usage
    exit 0
    ;;
  *)
    echo "Unknown argument: $1" >&2
    usage >&2
    exit 2
    ;;
esac

if [[ $# -gt 1 ]]; then
  echo "Too many arguments." >&2
  usage >&2
  exit 2
fi

TARGET_DIRS=(
  "${SH_TEST_DIR}/generated"
  "${SH_TEST_DIR}/results"
)

for target_dir in "${TARGET_DIRS[@]}"; do
  case "${target_dir}" in
    "${SH_TEST_DIR}/generated"|"${SH_TEST_DIR}/results")
      ;;
    *)
      echo "Refusing to clean unexpected path: ${target_dir}" >&2
      exit 1
      ;;
  esac

  if [[ -L "${target_dir}" ]]; then
    echo "Refusing to clean symbolic-link directory: ${target_dir}" >&2
    exit 1
  fi

  mkdir -p -- "${target_dir}"

  if [[ ${DRY_RUN} -eq 1 ]]; then
    echo "[dry-run] Contents to remove from ${target_dir}:"
    if ! find "${target_dir}" -mindepth 1 -maxdepth 1 -print -quit | grep -q .; then
      echo "  (empty)"
    else
      find "${target_dir}" -mindepth 1 -maxdepth 1 -print
    fi
    continue
  fi

  echo "[clean] Removing contents from ${target_dir} ..."
  find "${target_dir}" -mindepth 1 -delete
done

if [[ ${DRY_RUN} -eq 1 ]]; then
  echo "[dry-run] No files were removed."
else
  echo "[clean] Historical generated files and results have been removed."
fi
