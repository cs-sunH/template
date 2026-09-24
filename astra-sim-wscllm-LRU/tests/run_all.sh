#!/bin/bash
set -e

# Path
SCRIPT_DIR=$(dirname "$(realpath $0)")
PROJECT_DIR=$(realpath "${SCRIPT_DIR}/..")
export PYTHONPATH="${PROJECT_DIR}/extern/graph_frontend${PYTHONPATH:+:${PYTHONPATH}}"

echo "[$0] Running all regression tests..."

echo "[$0] Running WSC-LLM scheduler tests..."
(cd "${PROJECT_DIR}" && python3 -m unittest sh_test_mesh/workload/llama2_7b_inference/test_wsc_llm_scheduler.py) || (echo "Failed." ; exit 1)

echo "[$0] Running WSC-LLM KV incremental invariant tests..."
(cd "${PROJECT_DIR}" && python3 -m unittest sh_test_mesh/workload/llama2_7b_inference/test_wscllm_kv_incremental_invariants.py) || (echo "Failed." ; exit 1)

echo "[$0] Running WSC-LLM tiered eviction sequence tests..."
(cd "${PROJECT_DIR}" && python3 -m unittest sh_test_mesh/workload/llama2_7b_inference/test_tiered_eviction_sequence.py) || (echo "Failed." ; exit 1)

echo "[$0] Running WSC-LLM KV delta journal tests..."
(cd "${PROJECT_DIR}" && python3 -m unittest sh_test_mesh/workload/llama2_7b_inference/test_kv_delta_journal.py) || (echo "Failed." ; exit 1)

echo "[$0] Running online KV/scheduler unit tests..."
# unittest 套件走 -m unittest;pytest 风格(纯函数断言)套件直跑
# __main__ 入口(-m unittest 对它们发现 0 用例,恒绿)。
(cd "${PROJECT_DIR}" && python3 -m unittest sh_test_mesh/workload/llama2_7b_inference/online/test_admission_eviction_accumulation.py) || (echo "Failed." ; exit 1)
(cd "${PROJECT_DIR}" && python3 -m unittest sh_test_mesh/workload/llama2_7b_inference/online/test_admission_net_credit.py) || (echo "Failed." ; exit 1)
(cd "${PROJECT_DIR}" && python3 -m unittest sh_test_mesh/workload/llama2_7b_inference/online/test_decode_eviction_serialization.py) || (echo "Failed." ; exit 1)
(cd "${PROJECT_DIR}" && python3 -m unittest sh_test_mesh/workload/llama2_7b_inference/online/test_eviction_side_branch_structure.py) || (echo "Failed." ; exit 1)
(cd "${PROJECT_DIR}" && python3 -m unittest sh_test_mesh/workload/llama2_7b_inference/online/test_graph_batch_builder.py) || (echo "Failed." ; exit 1)
(cd "${PROJECT_DIR}" && python3 -m unittest sh_test_mesh/workload/llama2_7b_inference/online/test_kv_transfer_emission.py) || (echo "Failed." ; exit 1)
(cd "${PROJECT_DIR}" && python3 -m unittest sh_test_mesh/workload/llama2_7b_inference/online/test_store_restore_ordering.py) || (echo "Failed." ; exit 1)
(cd "${PROJECT_DIR}" && python3 -m unittest sh_test_mesh/workload/llama2_7b_inference/online/test_train_machinery.py) || (echo "Failed." ; exit 1)
(cd "${PROJECT_DIR}" && python3 sh_test_mesh/workload/llama2_7b_inference/online/test_graph_batch_rank_ledger.py) || (echo "Failed." ; exit 1)
(cd "${PROJECT_DIR}" && python3 sh_test_mesh/workload/llama2_7b_inference/online/test_propagating_tail.py) || (echo "Failed." ; exit 1)
(cd "${PROJECT_DIR}" && python3 sh_test_mesh/workload/llama2_7b_inference/online/test_weight_passes.py) || (echo "Failed." ; exit 1)

echo "[$0] Finished all regression tests."
