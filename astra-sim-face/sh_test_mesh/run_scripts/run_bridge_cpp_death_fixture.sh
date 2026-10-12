#!/usr/bin/env bash
# run_bridge_cpp_death_fixture.sh -- 缺陷 B 回归 fixture runner(2026-08-16
# Python 侧;2026-09-26 自母本 template/astra-sim-wscllm-LRU 同名脚本移植
# 到 face 仓:bridge_cpp_death_fixture.py 的 docstring 一直指名本脚本但
# face 仓此前漏带,自动断言链断裂)。
#
# 场景(纯 Python,无需 C++ 二进制/ET 物化):fixture fork 假 C++ 子进程
# 复刻 FileDecisionBridge 的 fd 语义后中途死亡 => 真实 BridgeServer 第
# 2 笔交付的 resp_notify 写 BrokenPipe => "C++ side is gone" stderr
# 留痕 + sys.exit(1)(fail-closed,不静默重试)。
#
# 断言(fixture docstring 契约):
#   1. fixture 退出码 == 1;
#   2. stderr 含 "C++ side is gone"。
#
# 用法:
#   bash run_bridge_cpp_death_fixture.sh [run_root]
set -uo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
PROJECT=$(realpath "${SCRIPT_DIR}/../..")
RUN_ROOT=${1:-/tmp/face_bridge_cpp_death}
# 相对 run_root 锚定调用者 cwd 预先解析为绝对路径（与 run_online_strategy.sh
# 同款，2026-09-29）：下方 cd 到工作负载目录后，fixture.out/.err 的相对基准
# 会从调用者 cwd 漂移到该目录——rm -rf 清错位置、产物散落两处。
case "${RUN_ROOT}" in /*) ;; *) RUN_ROOT="$(pwd -P)/${RUN_ROOT}" ;; esac
FIXTURE_REL=online/verify/bridge_cpp_death_fixture.py

rm -rf "${RUN_ROOT}"
mkdir -p "${RUN_ROOT}"

echo "[bridge_cpp_death] run root: ${RUN_ROOT}"

cd "${PROJECT}/sh_test_mesh/workload/llama2_7b_inference" || exit 1
fixture_exit=0
timeout 60 python3 "${FIXTURE_REL}" "${RUN_ROOT}/bridge" \
  > "${RUN_ROOT}/fixture.out" 2> "${RUN_ROOT}/fixture.err" \
  || fixture_exit=$?

echo "[bridge_cpp_death] fixture exit=${fixture_exit}"
if [[ ${fixture_exit} -ne 1 ]]; then
  echo "[bridge_cpp_death] FAIL: 期望退出码 1(BrokenPipe fail-closed),实际 ${fixture_exit}" >&2
  tail -20 "${RUN_ROOT}/fixture.err" >&2
  exit 1
fi
if ! grep -q "C++ side is gone" "${RUN_ROOT}/fixture.err"; then
  echo "[bridge_cpp_death] FAIL: stderr 缺 \"C++ side is gone\" 留痕" >&2
  tail -20 "${RUN_ROOT}/fixture.err" >&2
  exit 1
fi
echo "[bridge_cpp_death] PASS: C++ 中途死亡 => BrokenPipe fail-closed 退出 1 + stderr 留痕"
exit 0
