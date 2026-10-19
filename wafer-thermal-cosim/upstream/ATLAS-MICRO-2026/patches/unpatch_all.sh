#!/bin/bash

set -euo pipefail

echo "unpatch booksim2"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"${SCRIPT_DIR}/apply_patch_series.sh" "simulator/src/noc/booksim2" "patches/booksim2.patch"

echo "unpatch ramulator2"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"${SCRIPT_DIR}/apply_patch_series.sh" "simulator/src/dram/ramulator2" "patches/ramulator2.patch"

echo "unpatch tilelang"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"${SCRIPT_DIR}/apply_patch_series.sh" "frontend/tilelang" "patches/tilelang.patch"
