#!/bin/bash
# clean_test_records.sh —— 一键清除本仓全部仿真测试记录，还原裸仓库状态
# 用法: bash sh_test_mesh/run_scripts/clean_test_records.sh [--full]
#   默认: 删除全部运行产物/物化数据/缓存，git 恢复 trace_config.csv 原始字节
#   --full: 额外删除 build*/ 构建目录（下次需重新编译）
# 边界: 绝不触碰 git 跟踪文件（除用 git checkout 恢复 trace_config.csv）；
#       traces/ 仅保留 *.py 物化器脚本
# 退出码: 0=全部清除成功；1=存在清除失败项（结尾列出未清项清单）
set -u
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT" || exit 1
WL="sh_test_mesh/workload/llama2_7b_inference"
removed=0
failures=0
FAILED_ITEMS=()
# infra-E3(2026-09-30,失败传播):rm/find 的失败必须可见——此前 rm -rf 失败被
# 静默吞掉(计数照加、结尾照报"完成")。现 rm_art 记录失败条目并累加,任一
# 失败结尾非零退出并列出未清项。
rm_art() {
  if [ -e "$1" ]; then
    if rm -rf -- "$1"; then
      echo "[clean] removed: $1"
      removed=$((removed+1))
    else
      echo "[clean] FAILED to remove: $1 (rm exit=$?)" >&2
      FAILED_ITEMS+=("$1")
      failures=$((failures+1))
    fi
  fi
}

echo "[clean] repo: $REPO_ROOT"
# 1. 运行产物目录
rm_art "sh_test_mesh/generated"
rm_art "sh_test_mesh/results"
rm_art "sh_test_mesh/run_logs"
rm_art "log"
rm_art "sh_test_mesh/log"
rm_art "$WL/online_runs"
# X7(face-LRU 审计,2026-09-29):runner 在 workload 根落盘的后处理产物
# (raw/normalized_metrics.csv;历史版本曾把仓内副本 commit 进 git)。
rm_art "$WL/raw_metrics.csv"
rm_art "$WL/normalized_metrics.csv"
# X2(face-LRU 审计,2026-10-01):运行产物/物化件枚举补全——仓库根三件
# 指标产物(后处理在仓根 cwd 落盘)与 workload 根的 request_metrics.csv、
# request_queue_*.csv 及其 provenance、_canonical_sidecar*.csv 物化件。
# 通配经 find -maxdepth 1 收敛在 workload 根一层,不下钻 traces/
# (traces/ 物化数据由第 3 节按 *.py 白名单清理,物化器脚本保留);
# 占位指针恢复逻辑不读这些文件,git 恢复 trace_config.csv 不受影响。
rm_art "normalized_metrics.csv"
rm_art "raw_metrics.csv"
rm_art "request_metrics.csv"
rm_art "$WL/request_metrics.csv"
while read -r f; do rm_art "$f"; done \
  < <(find "$WL" -maxdepth 1 -type f \( -name "request_queue_*.csv" -o -name "request_queue_*.csv.provenance.json" -o -name "_canonical_sidecar*.csv" \) 2>/dev/null
       # (2026-10-01) derive 物化失败残留的 .part 临时件(仓根 cwd 缺省名运行时)
       find "$REPO_ROOT" -maxdepth 1 -type f -name "request_queue_*.csv.part" 2>/dev/null
       find "$REPO_ROOT" -maxdepth 1 -type f -name "_canonical_sidecar*.csv.part" 2>/dev/null)
# 2. completion_fixture 等测试再生目录
# infra-E3:改进程替换驱动循环——原 `find | while` 管道把循环体丢进子
# shell,rm_art 的失败计数/清单增量全部丢失。
while read -r d; do rm_art "$d"; done \
  < <(find sh_test_mesh -maxdepth 4 -type d -name "completion_fixture" 2>/dev/null)
# 3. traces/ 物化数据（仅保留 *.py 物化器脚本）
# infra-E3:去掉吞错的 2>/dev/null 并检查退出码——find -delete 半删失败
# 此前既不可见也不计数,结尾照报"已清除"。
if [ -d "$WL/traces" ]; then
  if ! find "$WL/traces" -mindepth 1 -type f ! -name "*.py" -delete; then
    echo "[clean] FAILED: find -delete (non-*.py data files) under $WL/traces" >&2
    FAILED_ITEMS+=("$WL/traces 内非 *.py 数据件")
    failures=$((failures+1))
  fi
  if ! find "$WL/traces" -mindepth 1 -type d -empty -delete; then
    echo "[clean] FAILED: find -delete (empty dirs) under $WL/traces" >&2
    FAILED_ITEMS+=("$WL/traces 内空目录")
    failures=$((failures+1))
  fi
  echo "[clean] traces/ 数据件已清除（仅保留 *.py 物化器脚本）"
fi
# 4. trace_config.csv 恢复 git 原始字节（若被仿真流程改指物化输入）
if git rev-parse --git-dir >/dev/null 2>&1; then
  # infra-E4(2026-09-30):git diff --quiet HEAD -- 联合对比 worktree+index——
  # 裸 `git diff --quiet --` 只比 worktree vs index,staged-only 修改(改后
  # git add 未 commit)检不出,占位态恢复被静默跳过。
  if ! git diff --quiet HEAD -- "$WL/trace_config.csv" 2>/dev/null; then
    # L21(2026-09-29 审查修复):git checkout HEAD -- 从 HEAD 提交恢复;
    # 裸 `git checkout --` 从 index 恢复——若 trace_config.csv 曾被 staged,
    # staged 差异不会被还原,且 HEAD 与 index 不一致时恢复的字节本身即错。
    if git checkout HEAD -- "$WL/trace_config.csv"; then
      echo "[clean] trace_config.csv 已恢复 git 原始字节"
    else
      echo "[clean] FAILED: git checkout HEAD -- $WL/trace_config.csv" >&2
      FAILED_ITEMS+=("$WL/trace_config.csv 恢复失败")
      failures=$((failures+1))
    fi
    echo "[clean] 提示: trace_config 已恢复占位态;重跑前需运行 traces/ 物化器脚本重新物化并将 request_queue_csv 指到物化产物（占位/缺失将 fail-closed）"
  fi
else
  echo "[clean][warn] 非 git 环境，跳过 trace_config 恢复（请手动核对 config,request_queue_csv 行指向 placeholder）"
fi
# 5. 缓存
# infra-E3:同上进程替换驱动 + 失败经 rm_art 计数(不再静默 rm)。
while read -r d; do rm_art "$d"; done \
  < <(find . -path ./build -prune -o -type d \( -name "__pycache__" -o -name ".pytest_cache" \) -print 2>/dev/null)
echo "[clean] __pycache__/.pytest_cache 已清除"
# 6. --full: 构建目录
if [ "${1:-}" = "--full" ]; then
  for b in build build_congestion_aware; do rm_art "$b"; done
fi
# infra-E3:结尾按失败计数分支——任一清除失败即非零退出并列出未清项,
# 不再无条件报"完成（还原裸仓库状态）"。
if [ "$failures" -gt 0 ]; then
  echo "[clean] FAIL: ${failures} 项清除失败，未清项清单:" >&2
  for item in "${FAILED_ITEMS[@]}"; do
    echo "[clean]   - ${item}" >&2
  done
  echo "[clean] 未完成（存在清除失败项，见上方清单）。当前 traces/ 内容: $(ls "$WL/traces" 2>/dev/null | tr '\n' ' ')" >&2
  exit 1
fi
echo "[clean] 完成（还原裸仓库状态）。当前 traces/ 内容: $(ls "$WL/traces" 2>/dev/null | tr '\n' ' ')"
