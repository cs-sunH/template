#!/bin/bash
set -e

# Path
SCRIPT_DIR=$(dirname "$(realpath $0)")
PROJECT_DIR=$(realpath "${SCRIPT_DIR}/..")
export PYTHONPATH="${PROJECT_DIR}/extern/graph_frontend${PYTHONPATH:+:${PYTHONPATH}}"

echo "[$0] Running all regression tests..."

echo "[$0] Running scheduler tests..."
(cd "${PROJECT_DIR}" && python3 -m unittest sh_test_mesh/workload/llama2_7b_inference/test_face_scheduler.py) || (echo "Failed." ; exit 1)

# ---------------------------------------------------------------------
# 嵌入热协同仿真（thermal_joint_integration_plan）：内嵌 wafercosim 包与
# joint 侧热桥接的 Python 回归。configs 作为 package data，故用包目录本身
# 作为 PYTHONPATH 即可，无需 pip 安装；离线可跑，不依赖 online binary。
# ---------------------------------------------------------------------
THERMAL_PYTHONPATH="${PROJECT_DIR}/extern/thermal_cosim:${PROJECT_DIR}/sh_test_mesh/workload/llama2_7b_inference"
echo "[$0] Running embedded thermal-core tests..."
(cd "${PROJECT_DIR}/extern/thermal_cosim" && PYTHONPATH="${THERMAL_PYTHONPATH}" python3 -m unittest discover -s tests -p 'test_*.py') || (echo "Failed." ; exit 1)
echo "[$0] Running joint thermal-bridge tests..."
(cd "${PROJECT_DIR}/sh_test_mesh/workload/llama2_7b_inference" && PYTHONPATH="${THERMAL_PYTHONPATH}" python3 -m unittest joint.test_thermal_bridge joint.test_thermal_hardware_export) || (echo "Failed." ; exit 1)
echo "[$0] Running online thermal wiring tests..."
(cd "${PROJECT_DIR}/sh_test_mesh/workload/llama2_7b_inference" && PYTHONPATH="${THERMAL_PYTHONPATH}" python3 -m unittest online.test_thermal_service_wiring) || (echo "Failed." ; exit 1)

# ---------------------------------------------------------------------
# 可选 C++ 段（H9 修复批，2026-09-29，与 face-LRU 仓 run_all.sh 同构）：
# MetricCollector 回归可执行目标此前从未被本入口执行。当且仅当仓内已
# 存在配置好的 build/astra_analytical/build_congestion_aware 构建树时，
# 增量构建并运行两个 MetricCollector 测试目标；构建树不存在则打印 SKIP
# （无 build 环境本脚本仍全绿可用），构建/运行失败则按 set -e 非零退出。
# 目标名见 astra-sim/network_frontend/analytical/CMakeLists.txt（grep
# one_shot / anchor_spool）。
# ---------------------------------------------------------------------
CPP_BUILD_DIR="${PROJECT_DIR}/build/astra_analytical/build_congestion_aware"
CPP_TEST_TARGETS=(
  AstraSim_Analytical_Congestion_Aware_MetricOneShotEraseTest
  AstraSim_Analytical_Congestion_Aware_MetricAnchorSpoolTest
)
if [ -f "${CPP_BUILD_DIR}/CMakeCache.txt" ]; then
  for cpp_target in "${CPP_TEST_TARGETS[@]}"; do
    echo "[$0] Building C++ target ${cpp_target} ..."
    cmake --build "${CPP_BUILD_DIR}" --target "${cpp_target}" -j 4
    cpp_bin=$(find "${CPP_BUILD_DIR}" -name "${cpp_target}" -type f -perm -u+x | head -n 1)
    if [ -z "${cpp_bin}" ]; then
      echo "[$0] C++ target built but binary not found: ${cpp_target}"
      exit 1
    fi
    echo "[$0] Running C++ target ${cpp_target} ..."
    "${cpp_bin}"
  done
else
  echo "[$0] SKIP C++ MetricCollector targets: no configured build tree at ${CPP_BUILD_DIR}"
fi

echo "[$0] Finished all regression tests."
