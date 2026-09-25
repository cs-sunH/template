#!/bin/bash
# clean_test_records.sh —— 一键清除本仓全部仿真测试记录，还原裸仓库状态
# 用法: bash sh_test_mesh/run_scripts/clean_test_records.sh [--full]
#   默认: 删除全部运行产物/物化数据/缓存，git 恢复 trace_config.csv 原始字节
#   --full: 额外删除 build*/ 构建目录（下次需重新编译）
# 边界: 绝不触碰 git 跟踪文件（除用 git checkout 恢复 trace_config.csv）；
#       traces/ 仅保留 *.py 物化器脚本
set -u
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT" || exit 1
WL="sh_test_mesh/workload/llama2_7b_inference"
rm_art() { if [ -e "$1" ]; then rm -rf "$1"; echo "[clean] removed: $1"; fi; }

echo "[clean] repo: $REPO_ROOT"
# 1. 运行产物目录
rm_art "sh_test_mesh/generated"
rm_art "sh_test_mesh/results"
rm_art "sh_test_mesh/run_logs"
rm_art "log"
rm_art "sh_test_mesh/log"
# 2. completion_fixture 等测试再生目录
find sh_test_mesh -maxdepth 4 -type d -name "completion_fixture" 2>/dev/null | while read -r d; do rm_art "$d"; done
# 3. traces/ 物化数据（仅保留 *.py 物化器脚本）
if [ -d "$WL/traces" ]; then
  find "$WL/traces" -mindepth 1 -type f ! -name "*.py" -delete 2>/dev/null
  find "$WL/traces" -mindepth 1 -type d -empty -delete 2>/dev/null
  echo "[clean] traces/ 数据件已清除（仅保留 *.py 物化器脚本）"
fi
# 4. trace_config.csv 恢复 git 原始字节（若被仿真流程改指物化输入）
if git rev-parse --git-dir >/dev/null 2>&1; then
  if ! git diff --quiet -- "$WL/trace_config.csv" 2>/dev/null; then
    git checkout -- "$WL/trace_config.csv" && echo "[clean] trace_config.csv 已恢复 git 原始字节"
  echo "[clean] 提示: trace_config 已恢复占位态;重跑前需运行 traces/ 物化器脚本重新物化并将 request_queue_csv 指到物化产物（占位/缺失将 fail-closed）"
  fi
else
  echo "[clean][warn] 非 git 环境，跳过 trace_config 恢复（请手动核对第12行指向 placeholder）"
fi
# 5. 缓存
find . -path ./build -prune -o -type d \( -name "__pycache__" -o -name ".pytest_cache" \) -print 2>/dev/null | while read -r d; do rm -rf "$d"; done
echo "[clean] __pycache__/.pytest_cache 已清除"
# 6. --full: 构建目录
if [ "${1:-}" = "--full" ]; then
  for b in build build_congestion_aware; do rm_art "$b"; done
fi
echo "[clean] 完成（还原裸仓库状态）。当前 traces/ 内容: $(ls "$WL/traces" 2>/dev/null | tr '\n' ' ')"
