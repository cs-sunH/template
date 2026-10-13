#!/usr/bin/env bash
set -euo pipefail

repo_root=/workspace/template
output_root="$repo_root/work/performance-current"
build_dir="$repo_root/work/build-die"
compiler=/usr/bin/c++
source_digest=e877a4ddc2dd5e0eff332fc8c01fcc33cc3ea92cfb8aac99c856b45a24f48b7a
verification_record="$repo_root/work/integration-build/verification-e877-final.json"
fixture_names=(gemm_1x128x128 gemm_128x128x128 attention_small_span)

cd "$repo_root"
mkdir -p "$output_root/raw"
exec > >(tee -a "$output_root/run.log") 2>&1
{
  printf 'timestamp_utc='; date -u '+%Y-%m-%dT%H:%M:%SZ'
  printf 'git_head='; git rev-parse HEAD
  printf 'published_source_digest=%s\n' "$source_digest"
  printf 'verification_record=%s\n' "$verification_record"
  if [[ -f "$verification_record" ]]; then sha256sum "$verification_record"; fi
  printf '\nworking_tree_status:\n'; git status --short
  printf '\ncompiler:\n'; "$compiler" --version
  printf '\ncmake:\n'; "$build_dir/../tooling/cmake/data/bin/cmake" --version 2>&1 || cmake --version
  printf '\nsystem:\n'; uname -a
  printf '\nloadavg_before_run:\n'; cat /proc/loadavg
  printf '\nprocessors:\n'; getconf _NPROCESSORS_ONLN
  if command -v lscpu >/dev/null 2>&1; then printf '\nlscpu:\n'; lscpu; fi
  if [[ -f "$build_dir/CMakeCache.txt" ]]; then
    printf '\nbuild_type_and_compiler:\n'
    rg '^(CMAKE_BUILD_TYPE|CMAKE_CXX_COMPILER|CMAKE_CXX_FLAGS_RELEASE):' "$build_dir/CMakeCache.txt" || true
  fi
  printf '\nbinary_sha256:\n'; sha256sum "$build_dir/coarse_die_benchmark" "$build_dir/lib/libcoarse_die_cabi.so" "$build_dir/libcoarse_die.a"
  printf '\nbinary_stat:\n'; stat "$build_dir/coarse_die_benchmark" "$build_dir/lib/libcoarse_die_cabi.so" "$build_dir/libcoarse_die.a"
  printf '\nproduction_source_sha256:\n'
  find "$repo_root/die/src" "$repo_root/die/include/coarse_die" -type f -print0 | sort -z | xargs -0 sha256sum
  sha256sum "$repo_root/die/CMakeLists.txt" "$repo_root/die/tests/benchmark.cc"
  printf '\ninput_sha256:\n'; sha256sum "$repo_root/die/tests/fixtures/coarse_die_v2.json" "$output_root/fixtures/"*.json
  printf '\nprobe_source_sha256:\n'; sha256sum "$output_root/cabi_cache_probe.cc" "$output_root/run_probes.sh"
} > "$output_root/environment.txt"

compile_command=("$compiler" -std=c++17 -O3 -DNDEBUG -I"$repo_root/die/include" -I"$repo_root/astra-sim-wscllm/extern/helper" "$output_root/cabi_cache_probe.cc" -L"$build_dir/lib" -Wl,-rpath,"$build_dir/lib" -lcoarse_die_cabi -o "$output_root/cabi_cache_probe")
{ printf 'compile: '; printf '%q ' "${compile_command[@]}"; printf '\n'; } > "$output_root/commands.txt"
"${compile_command[@]}"
sha256sum "$output_root/cabi_cache_probe" > "$output_root/cabi_cache_probe.sha256"

for name in "${fixture_names[@]}"; do
  benchmark_command=(python3 "$repo_root/die/tools/benchmark.py" --benchmark-bin "$build_dir/coarse_die_benchmark" --config "$repo_root/die/tests/fixtures/coarse_die_v2.json" --task "$output_root/fixtures/$name.json" --iterations 30 --owners 1 --output "$output_root/raw/cpp_compile_$name.json" --markdown "$output_root/raw/cpp_compile_$name.md")
  { printf 'cpp_benchmark: '; printf '%q ' "${benchmark_command[@]}"; printf '\n'; } >> "$output_root/commands.txt"
  "${benchmark_command[@]}"
  cache_command=("$output_root/cabi_cache_probe" --config "$repo_root/die/tests/fixtures/coarse_die_v2.json" --task "$output_root/fixtures/$name.json" --samples 30 --output "$output_root/raw/cabi_cache_$name.json")
  { printf 'cabi_cache: '; printf '%q ' "${cache_command[@]}"; printf '\n'; } >> "$output_root/commands.txt"
  "${cache_command[@]}"
done
{
  printf '\nloadavg_after_run:\n'; cat /proc/loadavg
  printf '\nprobe_binary_sha256:\n'; sha256sum "$output_root/cabi_cache_probe"
} >> "$output_root/environment.txt"

