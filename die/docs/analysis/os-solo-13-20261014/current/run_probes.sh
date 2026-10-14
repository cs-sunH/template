#!/usr/bin/env bash
set -euo pipefail

study_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "$study_dir/../../.." && pwd)"
build_dir="$repo_dir/work/build-die"
driver="$study_dir/os_lifecycle_runtime_driver"
log="$study_dir/run_probes.log"

exec > >(tee "$log") 2>&1
cd "$repo_dir"
source /workspace/cloud-setup/activate.sh

echo "[build] cmake --build $build_dir --target coarse_die -j2"
cmake --build "$build_dir" --target coarse_die -j2

echo "[compile] public Runtime driver; output remains inside study directory"
g++ -std=c++17 -Wall -Wextra -Werror -O2 \
  -I"$repo_dir/die/include" \
  -I"$repo_dir/astra-sim-wscllm/extern/helper" \
  -DCOARSE_DIE_SOURCE_DIR="\"$repo_dir/die\"" \
  "$repo_dir/die/tests/os_lifecycle_runtime_driver.cc" \
  "$build_dir/libcoarse_die.a" -o "$driver"

run_case() {
  local name="$1"
  shift
  echo "[facts] $name: $*"
  "$driver" "$@" > "$study_dir/facts_${name}.json"
}

run_case strict_small
run_case int4_native_small --path int4 --a-dtype int4 --w-dtype int4 --output-dtype int4
run_case int4_converted_fractional --path int4 --a-dtype bf16 \
  --kappa-quant 0.3125 --kappa-acc 0.375
run_case int4_converted_storage --path int4 --a-dtype bf16 --w-dtype int4 \
  --output-dtype int4 --kappa-quant 0.3125 --kappa-acc 0.375
run_case dense --m 65 --n 65 --k 4097 --cap 1
run_case large_k --m 65 --n 65 --k 64000001 --cap 1
run_case large_outputs --m 16000065 --n 65 --k 4097 --cap 1
run_case output_pressure_requested --m 129 --n 129 --k 2 \
  --tm 64 --tn 64 --tk 1
run_case output_pressure_k2_tiles128 --m 257 --n 257 --k 2 \
  --tm 128 --tn 128 --tk 1
run_case output_pressure_k1_tiles128 --m 257 --n 257 --k 1 \
  --tm 128 --tn 128 --tk 64

echo "[done] facts and producer binary are stored under $study_dir"
