#!/usr/bin/env bash
# run_python_unit_tests.sh —— sh_test_mesh Python 单测统一批量入口（2026-09-27
# 自 wscllm-LRU 同款移植，sched-D1 配套）。
#
# Usage: bash run_python_unit_tests.sh [extra pytest args...]
# 2026-09-29 (report 3.5 fix): the suite is NOT all unittest-style. Two test
# files are functional (bare `def test_*` functions + plain asserts): pytest
# collects them, `unittest discover` collects 0 from them, and a file
# without a `if __name__ == "__main__":` guard even exits 0 silently when
# direct-run. The runner therefore ALWAYS direct-runs the registered
# functional files (FUNCTIONAL_TESTS below) on top of the per-root sweep, in
# both modes, printing each file's case count and failing the whole entry on
# any nonzero exit. pytest missing (official image is "standard-library
# only", no pip pytest) still falls back to `python3 -m unittest discover`
# for the unittest-style suites; extra args only apply in pytest mode.
#
# 背景（旧报告 DC-孤儿测试测试接线）：6/7 KV/调度 Python 单测此前不在任何
# 回归入口；本脚本补上 Python 单测的单一清单入口，逐测试根调用 pytest 并
# 汇总退出码（任一根非 0 即整体非 0）。C++ 侧回归由 tests/run_all.sh 承担
# （2026-09-26 改造为 C++ ctest 统一入口、全量执行含 OfflineGreedy——该文件
# 实存且保留，并非"已删除"；两个入口的分工登记见根 README §5）。仍无 CI，
# 本入口只面向本地交付前回归。
#
# 测试根清单（与根 README §5 基线口径一致）：
#   1. workload/llama2_7b_inference（含 online/ 子目录）
#   2. tests/（sh_test_mesh/tests）
#   3. slo_tools/tests/
#
# 已知环境性结果（README §5，非本脚本缺陷）：
#   * （旧记「slo_tools/tests 1 例 LoadImbalance 手算环境性失败」已不复现——
#     2026-09-26 文档同步批复核实测 test_slo_contract.py 71/71 通过，删除该条）；
#   * trace_config.csv 的 request_queue_csv 指向真实队列时，
#     test_wsc_llm_scheduler 的 request-neutral 占位断言红（还原裸仓即绿）。
# 本脚本如实上报失败，不做任何排除/过滤（避免再现「自称全绿」的误导入口）。

set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SH_TEST_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

TEST_ROOTS=(
    "${SH_TEST_DIR}/workload/llama2_7b_inference"
    "${SH_TEST_DIR}/tests"
    "${SH_TEST_DIR}/slo_tools/tests"
)

# Functional test files (2026-09-29, report 3.5): bare `def test_*` +
# plain-assert files. `unittest discover` collects 0 cases from them (no
# TestCase subclass), and a file without a `__main__` guard even exits 0
# silently when executed -- so an unregistered functional file is silently
# dropped from this regression entry. This list is a registration mechanism,
# NOT an opt-in switch: the runner direct-runs EVERY file below in both
# modes, reports its top-level `def test_*` case count, and fails the whole
# entry on a nonzero exit or on a registered file that is missing / has
# zero cases. Adding a new functional test file REQUIRES registering it
# here.
FUNCTIONAL_TESTS=(
    "${SH_TEST_DIR}/workload/llama2_7b_inference/test_plan_materializer_passthrough.py"
    "${SH_TEST_DIR}/workload/llama2_7b_inference/online/test_propagating_tail.py"
)

overall_rc=0
# P3-3.9 E2(2026-09-28):官方镜像按 "standard-library only" 清理(根
# Dockerfile),pytest 不在镜像内——官方镜像 + 本入口的裸组合原样会在
# `python3 -m pytest` 处 "No module named pytest"。故 pytest 缺失时显式降级
# `python3 -m unittest discover` 运行 unittest 风格套件(逐根,同根同
# pass/fail 语义)。2026-09-29 修正:原注释"全部测试文件均为 unittest 风格,
# 与 pytest 同语义"不实——仓库存在函数式测试文件(unittest discover 收集数
# 为 0),它们由下方 FUNCTIONAL_TESTS 清单显式直跑兜底,不依赖 discover。
RUNNER_MODE=pytest
if ! python3 -m pytest --version >/dev/null 2>&1; then
    RUNNER_MODE=unittest
    echo "[run-python-unit-tests] NOTE: pytest not importable (official image is standard-library only); falling back to 'python3 -m unittest discover' for the unittest-style suites (functional files are direct-run separately, see FUNCTIONAL_TESTS). pytest remains the authoritative runner." >&2
fi
for root in "${TEST_ROOTS[@]}"; do
    if [ ! -d "${root}" ]; then
        echo "[run-python-unit-tests] SKIP（目录不存在）: ${root}"
        continue
    fi
    if [ "${RUNNER_MODE}" = "pytest" ]; then
        echo "[run-python-unit-tests] pytest ${root}"
        if ! python3 -m pytest -q "$@" "${root}"; then
            overall_rc=1
        fi
    else
        echo "[run-python-unit-tests] unittest discover ${root}"
        if ! python3 -m unittest discover -s "${root}" -t "${root}"; then
            overall_rc=1
        fi
    fi
done

# 2026-09-29 (report 3.5): explicit functional-file sweep, run in BOTH
# modes. In pytest mode these files are also collected by the root sweep
# above, but the direct run below is their authoritative self-check path
# (each file's documented `python3 <file>` entry) and keeps the
# FUNCTIONAL_TESTS registration exercised on every entry run.
functional_cases_total=0
for ftest in "${FUNCTIONAL_TESTS[@]}"; do
    if [ ! -f "${ftest}" ]; then
        echo "[run-python-unit-tests] FAIL（registered functional test file missing）: ${ftest}" >&2
        overall_rc=1
        continue
    fi
    case_count="$(grep -cE '^def test_' "${ftest}" || true)"
    echo "[run-python-unit-tests] functional ${ftest} (${case_count} case(s))"
    if [ "${case_count}" -eq 0 ]; then
        echo "[run-python-unit-tests] FAIL（registered functional test file exposes no top-level 'def test_*' case; stale or wrong registration）: ${ftest}" >&2
        overall_rc=1
        continue
    fi
    functional_cases_total=$((functional_cases_total + case_count))
    if ! python3 "${ftest}"; then
        overall_rc=1
    fi
done

if [ "${overall_rc}" -eq 0 ]; then
    echo "[run-python-unit-tests] 全部测试根通过 (runner=${RUNNER_MODE}, functional files=${#FUNCTIONAL_TESTS[@]}, functional cases=${functional_cases_total})"
else
    echo "[run-python-unit-tests] 存在失败测试根（见上方逐根输出）" >&2
fi
exit "${overall_rc}"
