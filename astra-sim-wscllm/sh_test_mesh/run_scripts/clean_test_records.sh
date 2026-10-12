#!/bin/bash
# clean_test_records.sh —— 一键清除本仓全部仿真测试记录，还原裸仓库状态
# 用法: bash sh_test_mesh/run_scripts/clean_test_records.sh [--full]
#   默认: 删除全部运行产物/物化数据/缓存，归一 trace_config.csv 的
#         request_queue_csv 指针行到占位指针（只改该行第 3 列，其余字节不动）
#   --full: 额外删除 build*/ 构建目录（下次需重新编译）
# 边界: 绝不触碰 git 跟踪文件（除归一 trace_config.csv 的指针行；
#       trace_config 相对 HEAD 的其余未提交修改只 WARN 列出、绝不回滚）；
#       traces/ 仅保留 *.py 物化器脚本
# (2026-10-03, die 推广批) 指针行归一替代原整文件 git checkout HEAD：
#   本仓 trace_config 自 die 批起携带未提交交付行（die system 模板 /
#   validation-die-64g 档 / position_encoding 行），整文件回滚会把交付
#   静默撤销——移植 face 仓同名脚本的 S-1/2026-10-01 语义收窄：只把
#   request_queue_csv 行第 3 列归一回 HEAD 版占位文件名。
set -u
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT" || exit 1
WL="sh_test_mesh/workload/llama2_7b_inference"
# (2026-10-01) The removed counter is gone: it was incremented inside a
# `while read` pipeline subshell (increment lost) and never read.
rm_art() { if [ -e "$1" ]; then rm -rf "$1"; echo "[clean] removed: $1"; fi; }

echo "[clean] repo: $REPO_ROOT"
# 1. 运行产物目录
rm_art "sh_test_mesh/generated"
rm_art "sh_test_mesh/results"
rm_art "sh_test_mesh/run_logs"
rm_art "log"
rm_art "sh_test_mesh/log"
rm_art "$WL/online_runs"
# 2. 仓根物化残留（traces/ 物化器 argv 裸文件名落 CWD 的历史形态；2026-09-29 RV6-F1：
#    只扫仓根 maxdepth 1 的普通文件、精确前缀匹配，不递归、不做宽泛 rm）
#    （crosscut-F17 2026-10-01：原第 2 节 completion_fixture 死清扫枚举已删——
#    生成器与消费者已于 2026-09-27 静态臂退役批删除，该项永空转）
find . -maxdepth 1 -type f -name "astra_compute_20_first_30_seconds_*" 2>/dev/null | while read -r f; do rm_art "$f"; done
# 3. traces/ 物化数据（仅保留 *.py 物化器脚本）
if [ -d "$WL/traces" ]; then
  find "$WL/traces" -mindepth 1 -type f ! -name "*.py" -delete 2>/dev/null
  find "$WL/traces" -mindepth 1 -type d -empty -delete 2>/dev/null
  echo "[clean] traces/ 数据件已清除（仅保留 *.py 物化器脚本）"
fi
# 4. trace_config.csv 恢复 git 原始字节（若被仿真流程改指物化输入）
# crosscut-F4（2026-10-01）：守卫与还原都改对 HEAD——git diff --quiet 只比
# 工作树 vs index，staged 后会静默跳过还原；git checkout -- 也从 index 还原，
# staged 版本 != HEAD 时结果不是原始字节。对齐 face-LRU / joint 同名脚本。
if git rev-parse --git-dir >/dev/null 2>&1; then
  # 指针行归一（2026-10-03 die 批起替代整文件 git checkout，见文件头注）：
  # 只把 request_queue_csv 行的第 3 列归一回 HEAD 版占位文件名，该行其余
  # 列与文件其余字节一律不动；相对 HEAD 的其余 diff 只 WARN 列出、绝不
  # 回滚。指针行无法归一（缺失/HEAD 无该行/改写失败）显式失败退出——
  # 静默继续会掩盖裸仓还原承诺被破坏。
  TC="$WL/trace_config.csv"
  HEAD_PTR=$(git show "HEAD:./$TC" 2>/dev/null | grep -m1 '^config,request_queue_csv,') || true
  WORK_PTR=$(grep -m1 '^config,request_queue_csv,' "$TC" 2>/dev/null) || true
  if [ ! -f "$TC" ] || [ -z "${WORK_PTR}" ]; then
    echo "[clean] FAIL: $TC 缺失或无 request_queue_csv 行，指针无法归一，裸仓状态不完整（请手工核对后重跑）" >&2
    exit 1
  fi
  if [ -z "${HEAD_PTR}" ]; then
    echo "[clean] FAIL: HEAD 版 $TC 无 request_queue_csv 行，无法归一指针，裸仓状态不完整" >&2
    exit 1
  fi
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
          n = 0; p2 = 0; p3 = 0
          for (i = 1; i <= length($0); i++) {
            if (substr($0, i, 1) == ",") {
              n++
              if (n == 2) p2 = i
              if (n == 3) { p3 = i; break }
            }
          }
          if (p3 == 0) print ENVIRON["FULLPTR"]
          else print substr($0, 1, p2) ENVIRON["REPL"] substr($0, p3)
        } else print }
    ' "$TC" > "$TMP" && chmod --reference="$TC" "$TMP" 2>/dev/null; then
      if mv -f "$TMP" "$TC"; then
        echo "[clean] trace_config.csv request_queue_csv 行已归一回占位指针（仅该行第 3 列，其余内容未触碰）"
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
    echo "[clean][WARN] trace_config.csv 相对 HEAD 仍有未提交差异（本脚本只归一了指针行、未回滚任何内容；可能为 die 批交付行或既有未提交修改，请自行核对后提交或处置）：" >&2
    git diff HEAD -- "$TC" >&2
  fi
  # R4-F1（2026-10-01）HEAD 自愈守卫（指针归一形态下为无害空转，保留）：
  # output_prefix/output_dir/record_planning_iterations 三零消费键若经任何
  # 路径复活（HEAD 回滚类操作）在此剥离；工作树无这三行时本条 no-op。
  sed -i '/^config,output_prefix,/d;/^config,output_dir,/d;/^config,record_planning_iterations,/d' \
    "$WL/trace_config.csv"
  echo "[clean] 提示: 重跑前需运行 traces/ 物化器脚本重新物化并将 request_queue_csv 指到物化产物（占位/缺失将 fail-closed）"
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
