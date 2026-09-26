#!/usr/bin/env bash
# run_python_unit_tests.sh —— sh_test_mesh Python 单测统一批量入口（2026-09-25）。
#
# Usage: bash run_python_unit_tests.sh [pytest 附加参数...]
#
# 背景（旧报告 DC-孤儿测试测试接线）：6/7 KV/调度 Python 单测此前不在任何
# 回归入口（误导性的 tests/run_all.sh 已删除）；本脚本补上单一清单入口，
# 逐测试根调用 pytest 并汇总退出码（任一根非 0 即整体非 0）。仍无 CI，
# 本入口只面向本地交付前回归。
#
# 测试根清单（与根 README §5 基线口径一致）：
#   1. workload/llama2_7b_inference（含 online/ 子目录，126 passed 基线）
#   2. tests/（sh_test_mesh/tests，49 passed 基线）
#   3. slo_tools/tests/（125 passed + 1 skipped 基线）
#
# 已知环境性结果（README §5，非本脚本缺陷）：
#   * slo_tools/tests 存在 1 例预存在环境性失败（LoadImbalance 手算×1）；
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

overall_rc=0
for root in "${TEST_ROOTS[@]}"; do
    if [ ! -d "${root}" ]; then
        echo "[run-python-unit-tests] SKIP（目录不存在）: ${root}"
        continue
    fi
    echo "[run-python-unit-tests] pytest ${root}"
    if ! python3 -m pytest -q "$@" "${root}"; then
        overall_rc=1
    fi
done

if [ "${overall_rc}" -eq 0 ]; then
    echo "[run-python-unit-tests] 全部测试根通过"
else
    echo "[run-python-unit-tests] 存在失败测试根（见上方逐根输出）" >&2
fi
exit "${overall_rc}"
