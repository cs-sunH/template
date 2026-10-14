#!/usr/bin/env bash
# Reconstruct the validated aggregate targets; binaries stay under work/.
set -euo pipefail
repro_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$repro_dir/../../../.." && pwd)"
if [[ -f /workspace/cloud-setup/activate.sh ]]; then
  source /workspace/cloud-setup/activate.sh
fi
unset PROTOBUF_FROM_SOURCE
scaffold="$repo_root/work/five-os-20261014/aggregate"
build_dir="$repo_root/work/lru-cpp-workspace-20261014/build/aggregate"
mkdir -p "$scaffold"
cp "$repro_dir/aggregate/CMakeLists.txt" "$scaffold/CMakeLists.txt"
cmake -S "$scaffold" -B "$build_dir" \
  -DASTRA_SIM_ROOT="$repo_root/astra-sim-wscllm-LRU" \
  -DBUILDTARGET=congestion_aware -DBUILD_TESTING=ON \
  -DASTRA_SIM_BUILD_OFFLINE_GREEDY_SCHEDULE_TEST=ON -DCMAKE_BUILD_TYPE=Release
cmake --build "$build_dir" --parallel 2 --target \
  AstraSim_Analytical_WorkspaceAdmissionTest \
  AstraSim_Analytical_Congestion_Aware_DecisionMailboxTest \
  AstraSim_Analytical_Congestion_Aware_GraphBatchCommitterTest \
  AstraSim_Analytical_Congestion_Aware_ParsedBatchTest \
  AstraSim_Analytical_Congestion_Aware_LocalHbmBandwidthModelTest \
  AstraSim_Analytical_Congestion_Aware_LocalHbmModelTest \
  AstraSim_Analytical_Congestion_Aware_DieModelAdapterTest \
  AstraSim_Analytical_Congestion_Aware_DieExecutionLifecycleTest \
  AstraSim_Analytical_Congestion_Aware_LocalHbmDieMemoryTest \
  AstraSim_Analytical_Congestion_Aware_Online coarse_die_cabi die_model_cabi
ctest --test-dir "$build_dir" --output-on-failure \
  -R '(WorkspaceAdmission|DecisionMailbox|GraphBatchCommitter|ParsedBatch|LocalHbmBandwidthModel|LocalHbmModel|DieModelAdapter|DieExecutionLifecycle|LocalHbmDieMemory)Test$'
