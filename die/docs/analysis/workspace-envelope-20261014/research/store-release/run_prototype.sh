#!/usr/bin/env bash
set -euo pipefail

TASK_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
BASELINE_DIR="$TASK_DIR/baseline-src/sh_test_mesh/workload/llama2_7b_inference"
PROTOTYPE_DIR="$TASK_DIR/prototype-src/sh_test_mesh/workload/llama2_7b_inference"
DRIVER="$TASK_DIR/test_source_release_capacity_gate.py"

source /workspace/cloud-setup/activate.sh

for workload_dir in "$BASELINE_DIR" "$PROTOTYPE_DIR"; do
  (
    cd "$workload_dir"
    python3 online/test_store_restore_ordering.py
    python3 online/test_eviction_side_branch_structure.py
    python3 online/test_kv_transfer_emission.py
  )
done

export WSC_WORKLOAD_DIR="$PROTOTYPE_DIR"
python3 -m py_compile \
  "$PROTOTYPE_DIR/online/graph_batch_builder.py" \
  "$PROTOTYPE_DIR/generate_wsc_llm_trace.py" \
  "$PROTOTYPE_DIR/online/online_service.py"
python3 "$DRIVER"

export WSC_WORKLOAD_DIR="$BASELINE_DIR"
python3 "$DRIVER" --dump-off > "$TASK_DIR/baseline-off.json"
export WSC_WORKLOAD_DIR="$PROTOTYPE_DIR"
python3 "$DRIVER" --dump-off > "$TASK_DIR/prototype-off.json"
cmp "$TASK_DIR/baseline-off.json" "$TASK_DIR/prototype-off.json"
sha256sum "$TASK_DIR/baseline-off.json" "$TASK_DIR/prototype-off.json"
