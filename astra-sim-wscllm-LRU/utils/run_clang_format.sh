#!/bin/bash
set -e

## ******************************************************************************
## This source code is licensed under the MIT license found in the
## LICENSE file in the root directory of this source tree.
##
## Copyright (c) 2024 Georgia Institute of Technology
## ******************************************************************************

# *****************************************************************************
# !!! WARNING / 危险 —— DO NOT RUN WITHOUT A DEDICATED ALIGNMENT DECISION !!!
#
# The checked-in .clang-format is NOT consistent with the actual formatting
# of this repository (BasedOnStyle: LLVM + InsertBraces/InsertNewlineAtEOF
# etc. contradict the tree as it stands). A full `-i` run of this script
# would rewrite a large fraction of the first-party sources (audit dry-run
# 2026-09: 91/166 files, ~7.8k lines of diff), which would silently destroy
# the byte-exact frozen baselines this repo's fixtures/gates rely on.
#
# 当前仓内 .clang-format 与代码实况不一致，未经专项对齐决策（先出对齐
# 方案、重跑全量冒烟/对拍基线）不得运行本脚本；-i 全量运行会重写大量
# 文件并破坏逐字节对拍基线（实测 dry-run 91/166 文件、约 7786 行 diff）。
# *****************************************************************************

# find the absolute path to this script
SCRIPT_DIR=$(dirname "$(realpath "$0")")
PROJECT_DIR="${SCRIPT_DIR:?}/.."

# 摩擦闸（R3 复核补强）：头注警告"未经专项对齐决策不得运行"（全量 -i
# 会重写 91/166 文件、破坏逐字节对拍基线），但脚本此前可被无条件执行。
# 现要求显式 opt-in：环境变量 CLANG_FORMAT_CONFIRM=yes 或传 --yes。
if [[ "${1:-}" != "--yes" && "${CLANG_FORMAT_CONFIRM:-}" != "yes" ]]; then
    echo "[run_clang_format] refusing: full-tree clang-format -i rewrites" >&2
    echo "  first-party sources and breaks byte-frozen baselines." >&2
    echo "  Re-run with --yes (or CLANG_FORMAT_CONFIRM=yes) after a" >&2
    echo "  dedicated alignment decision." >&2
    exit 1
fi

# start:
echo "[ASTRA-sim] Formatting Codebase..."
echo ""

# format all first-party sources under astra-sim/ plus the root tests/
# directory (*.cc/*.hh); extern/ third-party code is left untouched.
# clang-format-18 is the pinned binary (see the Dockerfile): the shared
# .clang-format uses InsertNewlineAtEOF (>=16) and InsertBraces (>=15).
find "${PROJECT_DIR:?}"/astra-sim "${PROJECT_DIR:?}"/tests \( -name "*.cc" -o -name "*.hh" \) -exec \
    clang-format-18 -style=file -i {} \;

# finalize
echo ""
echo "[ASTRA-sim] Codebase formatting Done."
