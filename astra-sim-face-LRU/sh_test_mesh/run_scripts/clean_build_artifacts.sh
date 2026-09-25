#!/bin/bash
# clean_build_artifacts.sh —— 一键清除本仓编译产物（裸仓库定义件之一）
# 用法: bash sh_test_mesh/run_scripts/clean_build_artifacts.sh
# 边界: 只删除编译产物目录（gitignored），绝不触碰任何 git 跟踪文件与
#       源码；清除后下次仿真前需重新 cmake 配置 + Release 编译（数分钟）。
# 配套: 裸仓库完整还原 = clean_test_records.sh（默认模式，清物化输入）
#       + 本脚本（清编译产物）；或一步 clean_test_records.sh --full。
set -u
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT" || exit 1
removed=0
rm_art() { if [ -e "$1" ]; then rm -rf "$1"; echo "[clean-build] removed: $1"; removed=$((removed+1)); fi; }

echo "[clean-build] repo: $REPO_ROOT"
# 兼容两种历史布局的编译目录名（现行唯一布局为顶层 build/）
for b in build build_congestion_aware; do rm_art "$b"; done
if [ "$removed" -eq 0 ]; then
  echo "[clean-build] 无编译产物（本仓已是裸仓库编译态）"
fi
echo "[clean-build] 完成（编译产物已清除）。"
