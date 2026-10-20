#!/bin/bash
# clean_test_records.sh —— 一键清除本仓全部仿真测试记录，还原裸仓库状态
# 用法: bash sh_test_mesh/run_scripts/clean_test_records.sh [--full]
#   默认: 删除全部运行产物/物化数据/缓存，trace_config.csv 仅归一指针行（不整文件回滚）
#   --full: 额外删除 build*/ 构建目录（下次需重新编译）
# 边界: 绝不触碰 git 跟踪文件（trace_config.csv 仅归一指针行第 3 列，不整文件回滚）；
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
# 占位指针归一逻辑不读这些文件，trace_config.csv 指针行归一不受影响。
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
# 4. trace_config.csv 指针行归一（若被仿真流程改指物化输入）
if git rev-parse --git-dir >/dev/null 2>&1; then
  # S-1 语义对齐（2026-10-20，对齐 face 仓变体）：裸仓定义第 1 条只要求
  # request_queue_csv 行指向占位串——本步只把该行第 3 列归一回 HEAD 版
  # 指针行的占位文件名（该行其余列与文件其余字节一律不动），不再整文件
  # git checkout HEAD：那会把作者未提交的非指针类修改（如容量默认档
  # validation-coarse-144g 的出厂翻转值）一并静默回滚（2026-10-20 实录：
  # 本脚本旧语义在 roofline 第三阶段收尾时误回滚三仓未提交默认档，经
  # 9d1767cc 补齐）。归一后该文件相对 HEAD 若仍有其余内容 diff，大声
  # WARN 列出并继续（由人提交或处置，绝不代为回滚、不以失败退出）；
  # 指针行本身无法归一（HEAD 无指针行/文件缺失/改写失败）仍显式失败
  # 退出——指针未还原时"裸仓还原"承诺已被破坏，静默继续会掩盖它。
  # 已登记边界（真仓格式不可触发，接受不加固）：归一判定只读首个指针行
  # （改写时 awk 会替换全部指针行；重复指针行的漂移可逃过判定、仅经
  # WARN 可见）；awk 重写给末行无换行的文件补一个换行（经 WARN 如实
  # 列出）。
  TC="$WL/trace_config.csv"
  HEAD_PTR=$(git show "HEAD:./$TC" 2>/dev/null | grep -m1 '^config,request_queue_csv,') || true
  WORK_PTR=$(grep -m1 '^config,request_queue_csv,' "$TC" 2>/dev/null) || true
  if [ ! -f "$TC" ] || [ -z "${WORK_PTR}" ]; then
    echo "[clean] FAIL: $TC 缺失或无 request_queue_csv 行，指针无法归一，裸仓状态不完整（请手工 git 恢复该文件后重跑）" >&2
    exit 1
  fi
  if [ -z "${HEAD_PTR}" ]; then
    echo "[clean] FAIL: HEAD 版 $TC 无 request_queue_csv 行，无法归一指针，裸仓状态不完整" >&2
    exit 1
  fi
  # 占位文件名 = HEAD 指针行第 3 列；含逗号/引号即超出"逐逗号定位第 3
  # 列"的解析模型，fail-closed 交人工。
  HEAD_VAL=$(printf '%s\n' "${HEAD_PTR}" | cut -d, -f3)
  case "${HEAD_VAL}" in
    ""|*","*|*'"'*)
      echo "[clean] FAIL: HEAD 指针行第 3 列非普通文件名（'${HEAD_VAL}'），拒绝盲改，请人工核对" >&2
      exit 1
      ;;
  esac
  if [ "$(printf '%s\n' "${WORK_PTR}" | cut -d, -f3)" != "${HEAD_VAL}" ]; then
    TMP="${TC}.normalize.$$.tmp"
    if REPL="${HEAD_VAL}" FULLPTR="${HEAD_PTR}" awk '
      { if ($0 ~ /^config,request_queue_csv,/) {
          # 只替换第 3 个逗号字段（指针文件名），其余列原样保留。
          n = 0; p2 = 0; p3 = 0
          for (i = 1; i <= length($0); i++) {
            if (substr($0, i, 1) == ",") {
              n++
              if (n == 2) p2 = i
              if (n == 3) { p3 = i; break }
            }
          }
          if (p3 == 0) print ENVIRON["FULLPTR"]   # 截断行（不足 3 列）整行回退 HEAD 指针行
          else print substr($0, 1, p2) ENVIRON["REPL"] substr($0, p3)
        } else print }
    ' "$TC" > "$TMP" && chmod --reference="$TC" "$TMP" 2>/dev/null; then
      if mv -f "$TMP" "$TC"; then
        echo "[clean] trace_config.csv request_queue_csv 行已归一回占位指针（仅该行第 3 列，其余内容未触碰）"
        echo "[clean] 提示: 重跑前需运行 traces/ 物化器脚本重新物化并将 request_queue_csv 指到物化产物（占位/缺失将 fail-closed）"
      else
        rm -f "$TMP"
        echo "[clean] FAIL: 指针行归一结果无法落盘（mv 失败），裸仓状态不完整" >&2
        exit 1
      fi
    else
      rm -f "$TMP"
      echo "[clean] FAIL: 指针行归一改写失败，裸仓状态不完整" >&2
      exit 1
    fi
  fi
  if ! git diff --quiet HEAD -- "$TC" 2>/dev/null; then
    echo "[clean][WARN] trace_config.csv 相对 HEAD 仍有未提交差异（本脚本只归一了指针行、未回滚任何内容；差异可能为既有未提交修改，也可能含脚本无法安全归一的畸形指针行，请自行核对后提交或处置）：" >&2
    git diff HEAD -- "$TC" >&2
  fi
else
  echo "[clean][warn] 非 git 环境，跳过 trace_config 指针归一（请手动核对 config,request_queue_csv 行指向 placeholder）"
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
