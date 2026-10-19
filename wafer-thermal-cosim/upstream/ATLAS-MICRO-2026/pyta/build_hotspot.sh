#!/usr/bin/env bash

set -Eeuo pipefail

pyta_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
hotspot_dir="${pyta_dir}/thirdparty/hotspot"
ignore_rules="${pyta_dir}/hotspot.gitignore"

if [[ ! -f "${hotspot_dir}/Makefile" ]] || \
   ! git -C "${hotspot_dir}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  printf 'HotSpot submodule is not initialized: %s\n' "${hotspot_dir}" >&2
  printf 'Run: git submodule update --init --recursive\n' >&2
  exit 1
fi

# A superproject .gitignore cannot apply inside a submodule. Install the
# repository-owned rules in the submodule's private Git metadata instead.
hotspot_git_dir=$(git -C "${hotspot_dir}" rev-parse --absolute-git-dir)
exclude_file="${hotspot_git_dir}/info/exclude"
marker='# ATLAS HotSpot build products (managed by pyta/build_hotspot.sh)'

mkdir -p "$(dirname -- "${exclude_file}")"
touch "${exclude_file}"
if ! grep -Fqx -- "${marker}" "${exclude_file}"; then
  printf '\n%s\n' "${marker}" >>"${exclude_file}"
fi

while IFS= read -r rule || [[ -n "${rule}" ]]; do
  case "${rule}" in
    ''|'#'*) continue ;;
  esac
  if ! grep -Fqx -- "${rule}" "${exclude_file}"; then
    printf '%s\n' "${rule}" >>"${exclude_file}"
  fi
done <"${ignore_rules}"

make -C "${hotspot_dir}" SUPERLU="${SUPERLU:-1}" "$@"
