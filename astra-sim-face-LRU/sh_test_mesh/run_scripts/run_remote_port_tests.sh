#!/bin/bash
# run_remote_port_tests.sh —— SerDes 片外链路并发化改造·远端端口三测试回归入口
# （方案阶段 5.7 口径：回归以本脚本的显式命令与退出码为准，ctest 不作为执行证据）
#
# 覆盖目标（CMake 全名，注册于 astra-sim/network_frontend/analytical/CMakeLists.txt）：
#   AstraSim_Analytical_Congestion_Aware_RemotePortNwayTest        —— 端口模型精确夹具
#       （PER_NPU/PER_NODE/MEMORY_POOL 映射、数学锚、RefPort oracle 交叉、
#         awaitcb 排除锚 134/101/151、tiny-residue clamp、sensing JSONL、
#         回调内同步重入（§3.3）、提前 shutdown（§3.4，幂等+删未交付
#         wlhd+复位计数）、fail-closed x12（含 undrained 析构反例）、
#         4000 流 stress；自含 mkdtemp，无需参数）
#   AstraSim_Analytical_Congestion_Aware_RemotePortOnlineGateTest  —— online 发射门控
#       （同 rank 双 MEM 单轮发射 issued/in_flight==2、MEM+COMM_SEND 双门共存、
#         peak_streaming>=2 与 shared_busy_ns>0 事件区间证据；无参数）
#   AstraSim_Analytical_Congestion_Aware_RemotePortStaticGateTest  —— static/ET 发射门控
#       （comm 单槽不挡第二 MEM、计数/集合恰释放一次、finish gate 等全部终结；
#         需 --fixture-dir <生成器产物目录>）
#
# 用法: bash sh_test_mesh/run_scripts/run_remote_port_tests.sh [--skip-build]
#   默认先按仓 README 完成过 configure 的既有 build 树增量构建三目标；
#   --skip-build 跳过构建直接运行（二进制须已存在）。
#
# 落地验证记录（2026-09-24，工作区=本批前端/后端改造合流后）：
#   构建命令（三目标逐个，均 exit 0）：
#     cmake --build build/astra_analytical/build_congestion_aware -j 8 \
#         --target AstraSim_Analytical_Congestion_Aware_RemotePortNwayTest
#     cmake --build build/astra_analytical/build_congestion_aware -j 8 \
#         --target AstraSim_Analytical_Congestion_Aware_RemotePortOnlineGateTest
#     cmake --build build/astra_analytical/build_congestion_aware -j 8 \
#         --target AstraSim_Analytical_Congestion_Aware_RemotePortStaticGateTest
#   fixture 生成（exit 0）：
#     python3 astra-sim/workload/execution_driven/tests/\
#         make_remote_port_static_fixture_et.py --out-dir <临时目录>
#   运行（fixture 参数 = 生成器 --out-dir 的同一目录）：
#     <bin>/...RemotePortNwayTest                                    -> exit 0
#     <bin>/...RemotePortOnlineGateTest                              -> exit 0
#     <bin>/...RemotePortStaticGateTest --fixture-dir <同上临时目录>   -> exit 0
#   三者 ALL PASS；nway 十个 phase（pernpup/anchor4/pernode/pool/zerolat/
#   awaitcb/tinyres/reentry/earlyshut/stress）逐个 ALL PASS，尾行含
#   "synchronous callback re-entry ... early shutdown ... fail-closed x12
#   ... 4000x10000B single-port stress"。
set -u
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT" || exit 1
BUILD=build/astra_analytical/build_congestion_aware
BIN="$BUILD/bin"
TESTS_DIR=astra-sim/workload/execution_driven/tests
TARGETS=(
    AstraSim_Analytical_Congestion_Aware_RemotePortNwayTest
    AstraSim_Analytical_Congestion_Aware_RemotePortOnlineGateTest
    AstraSim_Analytical_Congestion_Aware_RemotePortStaticGateTest
)

if [[ "${1:-}" != "--skip-build" ]]; then
    for t in "${TARGETS[@]}"; do
        echo "[remote-port-regression] cmake --build $BUILD --target $t"
        cmake --build "$BUILD" -j 8 --target "$t" || {
            echo "[remote-port-regression] BUILD FAILED: $t"; exit 1; }
    done
fi

FIXTURE_DIR="$(mktemp -d /tmp/remote_port_static_fixture_XXXXXX)"
echo "[remote-port-regression] python3 $TESTS_DIR/make_remote_port_static_fixture_et.py --out-dir $FIXTURE_DIR"
python3 "$TESTS_DIR/make_remote_port_static_fixture_et.py" --out-dir "$FIXTURE_DIR" || {
    echo "[remote-port-regression] FIXTURE GENERATION FAILED"; exit 1; }

rc=0
run_one() {
    local name="$1"; shift
    echo "[remote-port-regression] RUN: $name"
    "$@" || { echo "[remote-port-regression] FAIL: $name (exit $?)"; rc=1; }
}
run_one "${TARGETS[0]}" timeout 600 "$BIN/${TARGETS[0]}"
run_one "${TARGETS[1]}" timeout 600 "$BIN/${TARGETS[1]}"
run_one "${TARGETS[2]}" timeout 600 "$BIN/${TARGETS[2]}" --fixture-dir "$FIXTURE_DIR"

rm -rf "$FIXTURE_DIR"
if [[ $rc -eq 0 ]]; then
    echo "[remote-port-regression] ALL PASS (exit codes 0)"
else
    echo "[remote-port-regression] FAILURES PRESENT"
fi
exit "$rc"
