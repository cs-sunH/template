#!/usr/bin/env bash
# bridge_race_stress_repro.sh -- 缺陷 B 复现装置(2026-08-16,
# face主动测试错误分析.md 缺陷 B)。
#
# 背景:FileDecisionBridge 旧协议在 resp_notify 通道上按交换开/关:
#   C++:  open(O_RDONLY|O_NONBLOCK) -> poll(POLLIN|POLLHUP) -> read(1) -> close
#   Py :  open(O_WRONLY 阻塞) -> write(1 字节) -> close
# 该握手存在一族无法从应用层消除的内核竞态:
#   变体 1(F1 签名): C++ 新开读端的 poll_wait 恰逢写端 1->0 关闭转移且
#     FIFO 缓冲空 => POLLHUP + read()==0 => "Python side crashed" 误杀
#     (写端进程存活);
#   变体 2: Python 端 pending 的 open(O_WRONLY) 被读端唤醒后、write 落地
#     前读端已 close => BrokenPipeError。
# 修复(双侧长连接,2026-08-16)后本装置所复刻的旧模式不再存在于正式代码,
# 本脚本作为"缺陷曾经真实存在"的可运行证据保留:它复刻旧 syscall 模式,
# 在时间预算内观察到变体 1 即 PASS(实测本机 145 次/2M 内触发,负载相关)。
#
# 用法: bash bridge_race_stress_repro.sh [预算秒数,缺省 60]
set -uo pipefail

BUDGET=${1:-60}
WORK=$(mktemp -d /tmp/bridge_race_repro.XXXXXX)
trap 'rm -rf "${WORK}"' EXIT

# ---- 复刻旧 C++ 模式(逐交换 open/poll/read/close)----
cat > "${WORK}/old_cpp.c" <<'EOF'
#include <errno.h>
#include <fcntl.h>
#include <poll.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>
static const char* F;
int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <fifo> [iters]\n", argv[0]); return 2; }
    F = argv[1];
    long budget_iters = (argc > 2) ? atol(argv[2]) : 100000000;
    for (long i = 0; i < budget_iters; ++i) {
        int fd = open(F, O_RDONLY | O_NONBLOCK);
        if (fd < 0) { perror("open"); return 2; }
        struct pollfd pfd = { fd, POLLIN | POLLHUP, 0 };
        int r = poll(&pfd, 1, -1);
        if (r < 0) { perror("poll"); return 2; }
        char b = 0;
        ssize_t n = read(fd, &b, 1);
        close(fd);
        if (n == 0) {
            fprintf(stderr,
                    "[repro] FALSE-CRASH at iter %ld: poll revents=%u "
                    "(POLLHUP=%u) read==0 while writer alive\n",
                    i, pfd.revents, (unsigned)(pfd.revents & POLLHUP) == POLLHUP);
            printf("[repro] REPRODUCED false 'Python side crashed' at iter %ld\n", i);
            return 3;  /* the F1 signature */
        }
    }
    printf("[repro] no false-crash within budget iters\n");
    return 4;
}
EOF

# ---- 复刻旧 Python 模式(逐交换 open/write/close;EPIPE 容错继续,
#      让互补变体不提前终结压测)----
cat > "${WORK}/old_py.py" <<'EOF'
import os, sys, time
F = sys.argv[1]
deadline = time.time() + float(sys.argv[2])
epipes = 0
while time.time() < deadline:
    try:
        fd = os.open(F, os.O_WRONLY)
        try:
            os.write(fd, b"\n")
        finally:
            os.close(fd)
    except BrokenPipeError:
        epipes += 1  # variant-2 of the same race family (observed, logged)
print("[repro] python writer done; broken_pipes(observed variant 2) =", epipes)
EOF

gcc -O2 -o "${WORK}/old_cpp" "${WORK}/old_cpp.c" || { echo "gcc failed" >&2; exit 2; }
mkfifo "${WORK}/resp.fifo"

# 写端先起(阻塞在首个 open(O_WRONLY) 上),读端晚 0.2s 启动完成首个
# 交换的配对;写端按预算秒数运行。
# 读端窗口取 BUDGET-1,严格落在写端存活窗内:若两端同预算,写端被预算
# 杀掉后其写侧 fd 全关,读端下一轮 open+poll 必得 POLLHUP+read==0,
# 被 old_cpp 误判为 F1 签名(return 3),使 PASS 恒真、NOTE 分支不可达。
READER_BUDGET=$(awk -v b="${BUDGET}" 'BEGIN { r = b - 1; if (r <= 0) exit 1; printf "%.3f\n", r }') \
  || { echo "[bridge_race_stress_repro] 预算需 > 1 秒(读端窗口 = 预算-1)" >&2; exit 2; }
timeout "${BUDGET}" python3 "${WORK}/old_py.py" "${WORK}/resp.fifo" "${BUDGET}" \
    > "${WORK}/py.out" 2>&1 &
PY_PID=$!
sleep 0.2

timeout "${READER_BUDGET}" "${WORK}/old_cpp" "${WORK}/resp.fifo" 100000000 \
    > "${WORK}/cpp.out" 2>&1
CPP_RC=$?
wait "${PY_PID}" 2>/dev/null
PY_PID=""

echo "--- cpp:"; cat "${WORK}/cpp.out"
echo "--- py:"; cat "${WORK}/py.out"

if [[ ${CPP_RC} -eq 3 ]]; then
  echo "[bridge_race_stress_repro] PASS: 旧协议竞态(假 EOF 误杀)在本机复现;" \
       "正式代码已改双侧长连接消除该窗口"
  exit 0
fi
echo "[bridge_race_stress_repro] NOTE: 预算内未触发(竞态负载相关);" \
     "可加大预算或用 taskset 收紧 CPU 提高触发率" >&2
exit 1
