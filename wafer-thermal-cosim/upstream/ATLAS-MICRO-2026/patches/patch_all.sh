#!/bin/bash

set -euo pipefail

echo "patch booksim2"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"${SCRIPT_DIR}/export_patch_series.sh" "simulator/src/noc/booksim2" "patches/booksim2.patch" "origin/master"

echo "patch ramulator2"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"${SCRIPT_DIR}/export_patch_series.sh" "simulator/src/dram/ramulator2" "patches/ramulator2.patch" "origin/main"

echo "patch tilelang"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"${SCRIPT_DIR}/export_patch_series.sh" "frontend/tilelang" "patches/tilelang.patch" "origin/main"
