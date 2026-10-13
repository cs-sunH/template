#!/usr/bin/env bash
set -euo pipefail

repo_root=/workspace/template
output_root="$repo_root/work/performance-cache-response"
build_dir="$repo_root/work/build-die"
compiler=/usr/bin/c++
source_summary_digest=05c3f91c59e4ac3d22b5147a12566d93902814aa77bbf3384a9b7a5468a9e755
fixture_names=(gemm_1x128x128 gemm_128x128x128 attention_small_span)

cd "$repo_root"
if [[ ! -d "$output_root" || -e "$output_root/run.log" || -e "$output_root/environment.txt" ]]; then
  echo "refusing to overwrite a non-fresh output directory: $output_root" >&2
  exit 2
fi
mkdir -p "$output_root/raw" "$output_root/preflight"
exec > >(tee "$output_root/run.log") 2>&1

source_manifest() {
  find "$repo_root/die/src" "$repo_root/die/include/coarse_die" -type f -print0 | sort -z | xargs -0 sha256sum
  sha256sum "$repo_root/die/CMakeLists.txt" "$repo_root/die/tests/benchmark.cc"
}

{
  printf 'timestamp_utc='; date -u '+%Y-%m-%dT%H:%M:%SZ'
  printf 'git_head='; git rev-parse HEAD
  printf 'source_summary_digest=%s\n' "$source_summary_digest"
  printf 'source_summary_digest_provenance=parent integration record; per-file SHA-256 manifest below is independently captured'
  printf '\nworking_tree_status_relevant_paths:\n'; git status --short -- die/src die/include/coarse_die die/CMakeLists.txt die/tests/benchmark.cc
  printf '\ncompiler:\n'; "$compiler" --version
  printf '\ncmake:\n'; "$build_dir/../tooling/cmake/data/bin/cmake" --version 2>&1 || true
  printf '\nsystem:\n'; uname -a
  printf '\nloadavg_before_run:\n'; cat /proc/loadavg
  printf '\nprocessors:\n'; getconf _NPROCESSORS_ONLN
  printf '\nCPU:\n'; lscpu
  printf '\nmemory_before_run:\n'; free -h
  printf '\nbuild_type_and_compiler:\n'; rg '^(CMAKE_BUILD_TYPE|CMAKE_CXX_COMPILER|CMAKE_CXX_FLAGS_RELEASE):' "$build_dir/CMakeCache.txt" || true
  printf '\nbinary_sha256_before_run:\n'; sha256sum "$build_dir/coarse_die_benchmark" "$build_dir/lib/libcoarse_die_cabi.so" "$build_dir/libcoarse_die.a"
  printf '\nbinary_stat:\n'; stat "$build_dir/coarse_die_benchmark" "$build_dir/lib/libcoarse_die_cabi.so" "$build_dir/libcoarse_die.a"
} > "$output_root/environment.txt"
source_manifest > "$output_root/source_manifest.before.txt"

compile_command=("$compiler" -std=c++17 -O3 -DNDEBUG -I"$repo_root/die/include" -I"$repo_root/astra-sim-wscllm/extern/helper" "$output_root/cabi_cache_probe.cc" -L"$build_dir/lib" -Wl,-rpath,"$build_dir/lib" -lcoarse_die_cabi -o "$output_root/cabi_cache_probe")
{ printf 'compile_probe: '; printf '%q ' "${compile_command[@]}"; printf '\n'; } > "$output_root/commands.txt"
"${compile_command[@]}"
sha256sum "$output_root/cabi_cache_probe" > "$output_root/cabi_cache_probe.sha256"

# Retain the original one-sample setup preflight, separately from the 30-sample measurements.
for name in "${fixture_names[@]}"; do
  preflight_command=("$output_root/cabi_cache_probe" --config "$repo_root/die/tests/fixtures/coarse_die_v2.json" --task "$output_root/fixtures/$name.json" --samples 1 --output "$output_root/preflight/cabi_cache_$name.json")
  usage_command=(python3 "$output_root/run_with_usage.py" --metadata "$output_root/preflight/time_$name.json" -- "${preflight_command[@]}")
  { printf 'preflight_cabi: '; printf '%q ' "${usage_command[@]}"; printf '\n'; } >> "$output_root/commands.txt"
  "${usage_command[@]}"
done

for name in "${fixture_names[@]}"; do
  benchmark_command=(python3 "$repo_root/die/tools/benchmark.py" --benchmark-bin "$build_dir/coarse_die_benchmark" --config "$repo_root/die/tests/fixtures/coarse_die_v2.json" --task "$output_root/fixtures/$name.json" --iterations 30 --owners 1 --output "$output_root/raw/cpp_compile_$name.json" --markdown "$output_root/raw/cpp_compile_$name.md")
  usage_command=(python3 "$output_root/run_with_usage.py" --metadata "$output_root/raw/time_cpp_$name.json" -- "${benchmark_command[@]}")
  { printf 'cpp_benchmark: '; printf '%q ' "${usage_command[@]}"; printf '\n'; } >> "$output_root/commands.txt"
  "${usage_command[@]}"
  cache_command=("$output_root/cabi_cache_probe" --config "$repo_root/die/tests/fixtures/coarse_die_v2.json" --task "$output_root/fixtures/$name.json" --samples 30 --output "$output_root/raw/cabi_cache_$name.json")
  usage_command=(python3 "$output_root/run_with_usage.py" --metadata "$output_root/raw/time_cabi_$name.json" -- "${cache_command[@]}")
  { printf 'cabi_cache: '; printf '%q ' "${usage_command[@]}"; printf '\n'; } >> "$output_root/commands.txt"
  "${usage_command[@]}"
done

source_manifest > "$output_root/source_manifest.after.txt"
{
  printf '\nloadavg_after_run:\n'; cat /proc/loadavg
  printf '\nmemory_after_run:\n'; free -h
  printf '\nbinary_sha256_after_run:\n'; sha256sum "$build_dir/coarse_die_benchmark" "$build_dir/lib/libcoarse_die_cabi.so" "$build_dir/libcoarse_die.a"
  printf '\nprobe_binary_sha256:\n'; sha256sum "$output_root/cabi_cache_probe"
  printf '\nsource_manifest_diff_status:\n'; if cmp -s "$output_root/source_manifest.before.txt" "$output_root/source_manifest.after.txt"; then echo unchanged; else echo CHANGED; fi
} >> "$output_root/environment.txt"
