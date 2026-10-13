#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/../.." && pwd)"
baseline_library="$script_dir/baseline/libcoarse_die.a"
expected_library_sha="e0864d0eba1402a39654e572afc23a624a667fe7e04952a64486d99d6a560a9a"
actual_library_sha="$(sha256sum "$baseline_library" | awk '{print $1}')"
if [[ "$actual_library_sha" != "$expected_library_sha" ]]; then
  printf 'Baseline archive SHA mismatch: %s\n' "$actual_library_sha" >&2
  exit 2
fi

cxx="${CXX:-g++}"
binary="$script_dir/grouped_allocator_driver"
hardware_config="$repo_root/die/configs/example_hardware.json"
"$cxx" -std=c++17 -O2 -Wall -Wextra \
  -I "$repo_root/die/include" \
  -I "$repo_root/astra-sim-wscllm/extern/helper" \
  "$script_dir/grouped_allocator_driver.cc" \
  "$baseline_library" -o "$binary" \
  > "$script_dir/compile.log" 2>&1

"$binary" "$hardware_config" > "$script_dir/rates.raw.json"
python3 "$script_dir/finalize_results.py" \
  --raw "$script_dir/rates.raw.json" \
  --output "$script_dir/runtime-rates.json" \
  --binary "$binary" \
  --source "$script_dir/grouped_allocator_driver.cc" \
  --static-library "$baseline_library" \
  --hardware-config "$hardware_config" \
  --repo-root "$repo_root" \
  --compiler "$cxx"
