#!/usr/bin/env bash
set -euo pipefail

scratch_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "$scratch_dir/../.." && pwd)"
cd "$repo_dir"
source /workspace/cloud-setup/activate.sh

{
  echo "[compile] C++17 bounded Solo13 scratch kernel"
  g++ -std=c++17 -Wall -Wextra -Werror -O2 \
    -I"$repo_dir/astra-sim-wscllm-LRU/extern/helper" \
    -I"$repo_dir/die/include" \
    "$scratch_dir/solo13_kernel.cc" -o "$scratch_dir/solo13_kernel"
  echo "[check] exact Fraction and independent-prefix differential suite"
  python3 "$scratch_dir/check_kernel.py"
} 2>&1 | tee "$scratch_dir/build_and_check.log"
