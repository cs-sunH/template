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
removed=0
rm_art() { if [ -e "$1" ]; then rm -rf "$1"; echo "[clean] removed: $1"; removed=$((removed+1)); fi; }

echo "[clean] repo: $REPO_ROOT"
# 1. 运行产物目录
#    （W11 复核注记 2026-09-29）rm_art "sh_test_mesh/generated" 整树移除，
#    已覆盖全部三个已知夹具子目录；历次"clean 后仍在"并非枚举缺口，而是
#    clean 之后又被下列再生源重建，收尾时 clean 必须放在最后一次验证之后：
#    a. generated/runtime_config/<硬件slug>__<档>__<池> —— 规范口径广覆盖
#       pytest 的 workload 批次每次再生（test_face_scheduler 等经
#       load_face_trace_config → materialize_runtime_configs，输出目录由
#       SH_TEST_DIR 无条件推导，与 config csv 位置无关）；正式 runner 流程
#       同样落此处（产物根须在 clean 枚举根内，不改 /tmp）。
#    b. generated/completion_fixture —— astra-sim/workload/execution_driven/
#       tests/make_completion_fixture_et.py 显式生成（C++ 夹具前置）。
#    c. generated/hbm_nway_fixture —— 同上，make_hbm_nway_fixture.py 生成。
rm_art "sh_test_mesh/generated"
rm_art "sh_test_mesh/results"
rm_art "sh_test_mesh/run_logs"
rm_art "log"
rm_art "sh_test_mesh/log"
rm_art "$WL/online_runs"
rm -f "sh_test_mesh/runs/.single_simulation.lock" 2>/dev/null
# 2. completion_fixture 等测试再生目录（兜底：即使 generated/ 枚举行被
#    改动，夹具目录也按名清扫；hbm_nway_fixture 同族）
find sh_test_mesh -maxdepth 4 -type d \( -name "completion_fixture" -o -name "hbm_nway_fixture" \) 2>/dev/null | while read -r d; do rm_art "$d"; done
# 3. traces/ 物化数据（仅保留 *.py 物化器脚本）
if [ -d "$WL/traces" ]; then
  find "$WL/traces" -mindepth 1 -type f ! -name "*.py" -delete 2>/dev/null
  find "$WL/traces" -mindepth 1 -type d -empty -delete 2>/dev/null
  echo "[clean] traces/ 数据件已清除（仅保留 *.py 物化器脚本）"
fi
# 4. trace_config.csv 恢复 git 原始字节（若被仿真流程改指物化输入）
if git rev-parse --git-dir >/dev/null 2>&1; then
  if ! git diff --quiet -- "$WL/trace_config.csv" 2>/dev/null; then
    # L21(2026-09-29 审查修复):git checkout HEAD -- 从 HEAD 提交恢复;
    # 裸 `git checkout --` 从 index 恢复——若 trace_config.csv 曾被 staged,
    # staged 差异不会被还原,且 HEAD 与 index 不一致时恢复的字节本身即错。
    git checkout HEAD -- "$WL/trace_config.csv" && echo "[clean] trace_config.csv 已恢复 git 原始字节"
  echo "[clean] 提示: trace_config 已恢复占位态;重跑前需运行 traces/ 物化器脚本重新物化并将 request_queue_csv 指到物化产物（占位/缺失将 fail-closed）"
  fi
else
  echo "[clean][warn] 非 git 环境，跳过 trace_config 恢复（请手动核对 trace_config.csv 的 request_queue_csv 指向 placeholder）"
fi
# 5. 缓存
find . -path ./build -prune -o -type d \( -name "__pycache__" -o -name ".pytest_cache" \) -print 2>/dev/null | while read -r d; do rm -rf "$d"; done
echo "[clean] __pycache__/.pytest_cache 已清除"
# 6. --full: 构建目录
if [ "${1:-}" = "--full" ]; then
  for b in build build_congestion_aware; do rm_art "$b"; done
fi
echo "[clean] 完成（还原裸仓库状态）。当前 traces/ 内容: $(ls "$WL/traces" 2>/dev/null | tr '\n' ' ')"
