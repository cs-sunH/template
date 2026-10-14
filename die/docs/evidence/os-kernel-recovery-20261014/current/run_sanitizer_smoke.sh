#!/usr/bin/env bash
set -euo pipefail
scratch_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(git -C "$scratch_dir" rev-parse --show-toplevel)"
cd "$repo_dir"
g++ -std=c++17 -Wall -Wextra -Werror -O1 -g -fno-omit-frame-pointer \
  -fsanitize=address,undefined \
  -I"$repo_dir/astra-sim-wscllm-LRU/extern/helper" \
  -I"$scratch_dir/os-fixedwidth-13-20261014" \
  "$scratch_dir/kernel_sanitizer_smoke.cc" -o "$scratch_dir/kernel_sanitizer_smoke"
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 \
UBSAN_OPTIONS=print_stacktrace=1:halt_on_error=1 \
  "$scratch_dir/kernel_sanitizer_smoke" \
  "$scratch_dir/facts_input/facts_dense.json" \
  "$scratch_dir/facts_input/facts_large_k.json" \
  > "$scratch_dir/sanitizer_smoke.json" \
  2> "$scratch_dir/sanitizer_smoke.log"
