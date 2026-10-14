#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"

python3 "$SCRIPT_DIR/nested_k_output.py" > /dev/null
g++ -std=c++17 -O2 -Wall -Wextra -Wconversion -Wshadow \
  -frounding-math -fno-fast-math -ffp-contract=off \
  "$SCRIPT_DIR/fixed_interval_affine.cpp" \
  -o "$SCRIPT_DIR/fixed_interval_affine"
"$SCRIPT_DIR/fixed_interval_affine" \
  "$SCRIPT_DIR/nested-k-output.affine" > "$SCRIPT_DIR/fixed-interval.out"
python3 "$SCRIPT_DIR/check_fixed_interval.py"
