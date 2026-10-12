#!/bin/bash
# clean_test_records.sh —— 一键清除本仓全部仿真测试记录，还原裸仓库状态
# 用法: bash sh_test_mesh/run_scripts/clean_test_records.sh [--full]
#   默认: 删除全部运行产物/物化数据/缓存，归一 trace_config.csv 的
#         request_queue_csv 指针行到占位指针（只改该行第 3 列，其余字节不动）
#   --full: 额外删除 build*/ 构建目录（下次需重新编译）
# 边界: 绝不触碰 git 跟踪文件（除归一 trace_config.csv 的指针行；
#       trace_config 相对 HEAD 的其余未提交修改只 WARN 列出、绝不回滚）；
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
  # (2026-10-01) 删除被 2>/dev/null 吞错后不再无条件宣布成功:复核残留
  # 非 .py 文件,有残留即 WARN(只读目录/句柄占用等异常形状可见)。
  leftover=$(find "$WL/traces" -mindepth 1 -type f ! -name "*.py" 2>/dev/null | head -n 5)
  if [ -n "$leftover" ]; then
    echo "[clean] WARN: traces/ 仍有非 .py 残留未删除:" >&2
    echo "$leftover" >&2
  else
    echo "[clean] traces/ 数据件已清除（仅保留 *.py 物化器脚本）"
  fi
fi
# 4. trace_config.csv 指针行归一（若被仿真流程改指物化输入）。裸仓定义第 1
#    条只要求 request_queue_csv 行指向占位串——本步只把该行第 3 列归一回
#    HEAD 版指针行的占位文件名（该行其余列与文件其余字节一律不动），不再
#    整文件 git checkout HEAD：那会把作者未提交的非指针类修改（如残留键
#    清理）一并静默回滚（S-1/2026-10-01 语义收窄）。归一后该文件相对 HEAD
#    若仍有其余内容 diff，大声 WARN 列出并继续（由人提交或处置，绝不代为
#    回滚、不以失败退出）；指针行本身无法归一（HEAD 无指针行/文件缺失/
#    改写失败）仍显式失败退出——指针未还原时"裸仓还原"的头注承诺已被
#    破坏，静默继续会掩盖它。已登记边界（真仓格式不可触发，接受不加固）：
#    归一判定只读首个指针行（改写时 awk 会替换全部指针行；重复指针行的
#    漂移可逃过判定、仅经 WARN 可见）；
#    awk 重写会给末行无换行的文件补一个换行（该字节差异经 WARN 如实列出）。
if git rev-parse --git-dir >/dev/null 2>&1; then
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
  # 占位文件名 = HEAD 指针行第 3 列；含逗号/引号即超出本脚本"逐逗号定位
  # 第 3 列"的解析模型，fail-closed 交人工（占位文件名是普通相对路径，
  # 正常永不触发）。
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
          # 只替换第 3 个逗号字段（指针文件名），其余列原样保留：
          # 前缀 = 行首到第 2 个逗号（含），后缀 = 第 3 个逗号起（含）。
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
  echo "[clean][warn] 非 git 环境，跳过 trace_config 指针归一（请手动核对 request_queue_csv 行指向 placeholder）"
fi
# 5. 缓存
find . -path ./build -prune -o -type d \( -name "__pycache__" -o -name ".pytest_cache" \) -print 2>/dev/null | while read -r d; do rm -rf "$d"; done
echo "[clean] __pycache__/.pytest_cache 已清除"
# 6. --full: 构建目录
if [ "${1:-}" = "--full" ]; then
  for b in build build_congestion_aware; do rm_art "$b"; done
fi
echo "[clean] 完成（还原裸仓库状态）。当前 traces/ 内容: $(ls "$WL/traces" 2>/dev/null | tr '\n' ' ')"
