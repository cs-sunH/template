#!/usr/bin/env bash
set -euo pipefail

scratch_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "$scratch_dir/../.." && pwd)"
cd "$repo_dir"
source /workspace/cloud-setup/activate.sh

g++ -std=c++17 -Wall -Wextra -Werror -O2 \
  -I"$repo_dir/astra-sim-wscllm-LRU/extern/helper" \
  -I"$repo_dir/die/include" \
  "$scratch_dir/solo13_benchmark.cc" -o "$scratch_dir/solo13_benchmark"

g++ -std=c++17 -Wall -Wextra -Werror -O2 -DSOLO13_PROFILE_EXACT \
  -I"$repo_dir/astra-sim-wscllm-LRU/extern/helper" \
  -I"$repo_dir/die/include" \
  "$scratch_dir/solo13_benchmark.cc" -o "$scratch_dir/solo13_benchmark_profile"

python3 "$scratch_dir/run_benchmarks.py"
