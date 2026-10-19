#!/bin/bash
# tests/run_all.sh —— C++ ctest 统一回归入口（2026-09-26 改造，M7-04/M7-09）。
# 原 Python 套件批跑职责移交 sh_test_mesh/run_scripts/run_python_unit_tests.sh
# （两个入口的分工登记见根 README §5）。
#
# 覆盖：canonical 聚合树 build/astra_analytical/build_congestion_aware 内
# 全部 add_test 注册用例，ctest 全量执行、不加名字过滤——
#   - AstraSim_OfflineGreedyScheduleJournalTest（根 CMakeLists 注册，恒在；
#     旧版 `ctest -R "Analytical_"` 匹配不到它，等于唯一常驻用例永不执行，
#     2026-09-26 去过滤修正）；
#   - Analytical_FluidSchedulerLinkObserverTest / Analytical_StaleEventCancellationTest
#     （挂在默认 OFF 的 ANALYTICAL_BUILD_*_TEST option 上，且只有
#     NETWORK_BACKEND_BUILD_AS_LIBRARY=ON 的 canonical 装配树才能链接；
#     沿用瞬开瞬关机制：重配置打开 -> 构建测试目标 -> ctest 全量 -> 还原 OFF，
#     测试目标不留在普通 all 构建图里）。
# 树缺席/配置不符时打印显式 SKIP（不静默），缺什么写进消息。
set -e

# Path（infra-E2 修正：$0 不加引号在含空格路径下会让 realpath 收到两个参数、
# PROJECT_DIR 错位 → 误判 canonical 树缺席而 SKIP 恒 exit 0；改用 BASH_SOURCE
# 引号保护 + pwd -P，对绝对/相对/符号链接调用均稳定）
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
PROJECT_DIR=$(realpath "${SCRIPT_DIR}/..")

ANALYTICAL_BUILD_DIR="${PROJECT_DIR}/build/astra_analytical/build_congestion_aware"
ANALYTICAL_CACHE="${ANALYTICAL_BUILD_DIR}/CMakeCache.txt"
if [ ! -f "${ANALYTICAL_CACHE}" ]; then
    echo "[$0] SKIP: C++ ctest regression -- canonical build tree not found at ${ANALYTICAL_BUILD_DIR} (README §2 step (1))"
    exit 0
fi

# infra-E2 修正：瞬开瞬关的 ANALYTICAL_BUILD_*_TEST 开关必须在任一退出路径上
# 还原 OFF——set -e 下"打开开关后的重配置/构建失败"会直接退出，绕过文件尾
# 的还原逻辑，测试目标从此滞留普通 all 构建图。用 EXIT trap 兜底（正常结束
# 也走同一条还原路径，替代原文件尾的手工还原块）。
TOGGLES_OPENED=0
restore_toggles() {
    if [ "${TOGGLES_OPENED}" -eq 1 ]; then
        TOGGLES_OPENED=0
        if (cd "${PROJECT_DIR}/build/astra_analytical" && cmake -S . -B build_congestion_aware \
                -DANALYTICAL_BUILD_FLUID_SCHEDULER_LINK_OBSERVER_TEST=OFF \
                -DANALYTICAL_BUILD_STALE_EVENT_CANCELLATION_TEST=OFF); then
            echo "[$0] ANALYTICAL_BUILD_*_TEST toggles restored to OFF"
        else
            echo "[$0][warn] failed to restore ANALYTICAL_BUILD_*_TEST toggles to OFF; inspect ${ANALYTICAL_CACHE}" >&2
        fi
    fi
}
trap restore_toggles EXIT

# OfflineGreedy 测试目标只依赖 AstraSim 静态库、不依赖后端库，先行构建，
# 保证下方 ctest 全量是真实执行而非"目标缺席"。
(cd "${ANALYTICAL_BUILD_DIR}" && cmake --build . --target \
    AstraSim_OfflineGreedyScheduleJournalTest -j) || (echo "Failed." ; exit 1)

if ! grep -q "^NETWORK_BACKEND_BUILD_AS_LIBRARY:BOOL=ON$" "${ANALYTICAL_CACHE}" || \
     ! grep -Eq "^BUILDTARGET:STRING=(congestion_aware|all)$" "${ANALYTICAL_CACHE}"; then
    echo "[$0] SKIP: analytical fluid regression tests -- ${ANALYTICAL_CACHE} lacks NETWORK_BACKEND_BUILD_AS_LIBRARY=ON or BUILDTARGET=congestion_aware|all"
else
    # 置位先于重配置：若 cmake -D ON 写入缓存后配置/构建才失败，trap 仍会还原。
    TOGGLES_OPENED=1
    (cd "${PROJECT_DIR}/build/astra_analytical" && cmake -S . -B build_congestion_aware \
        -DANALYTICAL_BUILD_FLUID_SCHEDULER_LINK_OBSERVER_TEST=ON \
        -DANALYTICAL_BUILD_STALE_EVENT_CANCELLATION_TEST=ON) || (echo "Failed." ; exit 1)
    (cd "${ANALYTICAL_BUILD_DIR}" && cmake --build . --target \
        Analytical_FluidSchedulerLinkObserverTest \
        Analytical_StaleEventCancellationTest -j) || (echo "Failed." ; exit 1)
fi

# M7-09（2026-09-26）：ctest 全量执行（--test-dir 指向 canonical 聚合树，
# 含子目录注册的全部用例），不再 -R "Analytical_" 过滤。
ctest --test-dir "${ANALYTICAL_BUILD_DIR}" --output-on-failure \
    || (echo "Failed." ; exit 1)

# 测试开关还原由 EXIT trap（restore_toggles）统一承担，正常/失败路径均覆盖。

echo "[$0] Finished all C++ ctest regression tests."
