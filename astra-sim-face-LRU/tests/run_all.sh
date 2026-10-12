#!/bin/bash
# tests/run_all.sh —— 本仓 Python 单元测试全量回归入口。
#
# 收编全部 19 个测试文件（相对仓根）：
#   sh_test_mesh/tests 3 + sh_test_mesh/slo_tools/tests 4 +
#   workload 根 3 + online/ 9。
# 逐文件运行（保持仓根 CWD 与 PYTHONPATH 导出）：unittest.TestCase 风格
# 的 16 个用 python3 -m unittest（与既有入口同款）；纯 pytest 风格的
# 3 个（plain-assert 函数、无 TestCase、unittest 收集为 0 例）用
# python3 -m pytest。任一文件失败即在本脚本退出码非零中体现；末尾汇总
# 失败清单。
#
# 基线（2026-09-26 实测复核；2026-10-01 修复批后实测 295 例）：19 个
#   测试文件全绿——tests/ 49 + workload 根 33 + online/ 77（unittest 62
#   + pytest 15）+ slo_tools/tests 136。test_hbm_watermark 的
#   JournalParityRealTests 依赖仓外 fixture /var/tmp/phase2_journal/runs/on_2s，
#   缺失时自动 skip（非失败）。历史注记：slo_tools/tests 曾有 3 例改前基线
#   既有失败（driver_parity ×2 + load_imbalance 手算例，原 face 仓与 sh_2.0 仓
#   同样失败），已由 2026-09-26 修复批处理，现测全部通过（见 README §5）。
# 可选 C++ 段（H9 修复批，2026-09-29）：此前本入口只跑 Python，MetricCollector
#   的两个回归可执行（one-shot erase / anchor spool）从未被任何入口执行，
#   其 oracle 腐烂无人知晓。现当且仅当仓内已存在配置好的
#   build/astra_analytical/build_congestion_aware 构建树时，增量构建并运行
#   这两个目标；构建树不存在则打印 SKIP（无 build 环境本脚本仍全绿可用），
#   构建树存在但构建/运行失败则如实计入失败清单。
set -u

# T6(face-LRU 审计,2026-09-29):内层 $0 必须加引号——绝对路径(含空格)
# 调用时词分割使 SCRIPT_DIR 变多行垃圾,cd 失败 exit 1。
SCRIPT_DIR=$(dirname "$(realpath "$0")")
PROJECT_DIR=$(realpath "${SCRIPT_DIR}/..")
export PYTHONPATH="${PROJECT_DIR}/extern/graph_frontend${PYTHONPATH:+:${PYTHONPATH}}"
cd "${PROJECT_DIR}" || exit 1

# unittest.TestCase 风格（python3 -m unittest 驱动）
PY_TESTS_UNITTEST=(
  sh_test_mesh/tests/test_config_resolver.py
  sh_test_mesh/tests/test_first_token_proxy.py
  sh_test_mesh/tests/test_metrics_contract.py
  sh_test_mesh/slo_tools/tests/test_driver_parity.py
  sh_test_mesh/slo_tools/tests/test_golden_g1g4.py
  sh_test_mesh/slo_tools/tests/test_hbm_watermark.py
  sh_test_mesh/slo_tools/tests/test_slo_contract.py
  sh_test_mesh/workload/llama2_7b_inference/test_face_kv_incremental_invariants.py
  sh_test_mesh/workload/llama2_7b_inference/test_face_scheduler.py
  sh_test_mesh/workload/llama2_7b_inference/test_face_tiered_eviction_sequence.py
  sh_test_mesh/workload/llama2_7b_inference/online/test_admission_liveness.py
  sh_test_mesh/workload/llama2_7b_inference/online/test_eviction_side_branch_structure.py
  sh_test_mesh/workload/llama2_7b_inference/online/test_first_token_split.py
  sh_test_mesh/workload/llama2_7b_inference/online/test_forward_edge_invariant.py
  sh_test_mesh/workload/llama2_7b_inference/online/test_graph_batch_builder.py
  sh_test_mesh/workload/llama2_7b_inference/online/test_store_restore_ordering.py
  sh_test_mesh/workload/llama2_7b_inference/online/test_train_machinery.py
  sh_test_mesh/workload/llama2_7b_inference/online/test_transfer_frontier_neutral.py
)

# 纯 pytest 风格（python3 -m pytest 驱动；unittest 收集为 0 例）
PY_TESTS_PYTEST=(
  sh_test_mesh/workload/llama2_7b_inference/online/test_graph_batch_rank_ledger.py
  sh_test_mesh/workload/llama2_7b_inference/online/test_propagating_tail.py
  sh_test_mesh/workload/llama2_7b_inference/online/test_weight_passes.py
)

total=$(( ${#PY_TESTS_UNITTEST[@]} + ${#PY_TESTS_PYTEST[@]} ))
failed=()
for test_path in "${PY_TESTS_UNITTEST[@]}"; do
  echo "[$0] Running ${test_path} ..."
  if ! python3 -m unittest "${test_path}"; then
    failed+=("${test_path}")
  fi
done
for test_path in "${PY_TESTS_PYTEST[@]}"; do
  echo "[$0] Running ${test_path} (pytest) ..."
  if ! python3 -m pytest -q "${test_path}"; then
    failed+=("${test_path}")
  fi
done

# ---------------------------------------------------------------------
# 可选 C++ 段（H9 修复批，2026-09-29）：MetricCollector 回归可执行目标。
# 目标名见 astra-sim/network_frontend/analytical/CMakeLists.txt（grep
# one_shot / anchor_spool）。仅在仓内已存在配置好的构建树时启用。
# ---------------------------------------------------------------------
CPP_BUILD_DIR="${PROJECT_DIR}/build/astra_analytical/build_congestion_aware"
CPP_TEST_TARGETS=(
  AstraSim_Analytical_Congestion_Aware_MetricOneShotEraseTest
  AstraSim_Analytical_Congestion_Aware_MetricAnchorSpoolTest
)
if [ -f "${CPP_BUILD_DIR}/CMakeCache.txt" ]; then
  for cpp_target in "${CPP_TEST_TARGETS[@]}"; do
    echo "[$0] Building C++ target ${cpp_target} ..."
    if ! cmake --build "${CPP_BUILD_DIR}" --target "${cpp_target}" -j 4; then
      failed+=("cpp-build:${cpp_target}")
      continue
    fi
    cpp_bin=$(find "${CPP_BUILD_DIR}" -name "${cpp_target}" -type f -perm -u+x | head -n 1)
    if [ -z "${cpp_bin}" ]; then
      echo "[$0] C++ target built but binary not found: ${cpp_target}"
      failed+=("cpp-binary:${cpp_target}")
      continue
    fi
    echo "[$0] Running C++ target ${cpp_target} ..."
    if ! "${cpp_bin}"; then
      failed+=("cpp-run:${cpp_target}")
    fi
  done
else
  echo "[$0] SKIP C++ MetricCollector targets: no configured build tree at ${CPP_BUILD_DIR}"
fi

if [ "${#failed[@]}" -gt 0 ]; then
  echo "[$0] FAILED test files (${#failed[@]}/${total}):"
  for test_path in "${failed[@]}"; do
    echo "[$0]   - ${test_path}"
  done
  exit 1
fi

echo "[$0] All ${total} Python test files passed."
