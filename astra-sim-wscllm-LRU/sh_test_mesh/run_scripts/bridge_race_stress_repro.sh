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
#include <time.h>
#include <unistd.h>
static const char* F;
int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <fifo> [iters] [marker] [ready_wait_s]\n", argv[0]); return 2; }
    F = argv[1];
    long budget_iters = (argc > 2) ? atol(argv[2]) : 100000000;
    const char* marker = (argc > 3) ? argv[3] : NULL;
    double ready_wait_s = (argc > 4) ? atof(argv[4]) : 30.0;
    /* R4 rendezvous: hold the FIRST read fd open and wait (stat loop) for
     * the writer's ready marker BEFORE the first poll. With this fd held,
     * the writer's non-blocking O_WRONLY open cannot get ENXIO, so the
     * marker is guaranteed to appear; polling before the writer's first
     * open would see POLLHUP+read==0 (no writer yet) -- the exact false
     * positive this fixture must not manufacture. The writer keeps its
     * first fd open briefly past the marker, so the first poll after this
     * point sees a live writer. */
    {
        int hfd = open(F, O_RDONLY | O_NONBLOCK);
        if (hfd < 0) { perror("open"); return 2; }
        if (marker != NULL) {
            time_t start = time(NULL);
            while (access(marker, F_OK) != 0) {
                if (difftime(time(NULL), start) > ready_wait_s) {
                    fprintf(stderr, "[repro] writer ready marker timeout\n");
                    close(hfd);
                    return 2;
                }
                usleep(2000);
            }
        }
        usleep(1000);
        close(hfd);
    }
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
import errno, os, sys, time
F = sys.argv[1]
deadline = time.time() + float(sys.argv[2])
MARKER = sys.argv[3]
ready = False
epipes = 0

def open_writer():
    # R4 rendezvous: non-blocking open + ENXIO retry. The OLD blocking
    # O_WRONLY open deadlocked the fixture: it waits for a reader, the
    # reader waits for this writer's ready marker, and the marker only
    # lands after the open returns. The reader now holds its first fd
    # while waiting for the marker, so this retry pairs immediately.
    while True:
        try:
            return os.open(F, os.O_WRONLY | os.O_NONBLOCK)
        except OSError as exc:
            if exc.errno == errno.ENXIO and time.time() < deadline:
                time.sleep(0.002)
                continue
            raise

while time.time() < deadline:
    try:
        fd = open_writer()
        if not ready:
            # L21(2026-09-29 审查修复,face-LRU 同款移植):阻塞式 O_WRONLY
            # open 返回即代表读端已与本写端配对;就绪标记只能在这之后落盘。
            # 读端侧据此轮询就绪再开跑,消除固定 sleep 0.2 的握手竞态(写端
            # 尚未起跑时读端首轮 POLLHUP+read==0 会被误判成变体 1 签名 =>
            # 假阳性 PASS)。
            with open(MARKER, "w") as m:
                m.write("ready\n")
            ready = True
            # Hold this first fd past the marker handoff (~50ms): the
            # reader polls only after seeing the marker; this fd
            # guarantees a live writer at that instant.
            time.sleep(0.05)
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

# L21(2026-09-29 审查修复,face-LRU 同款移植):读端改为轮询写端就绪标记
# (带超时)后再开跑,替代固定 sleep 0.2——慢机上写端尚未完成首个 O_WRONLY
# open 时就启动读端,首轮 poll 即得 POLLHUP+read==0(写端缺席,并非"写端
# 1->0 关闭转移"),会被误判成变体 1 签名 => 假阳性 PASS。写端在阻塞 open
# 配对成功后落就绪标记(见上方 old_py.py),轮询超时上限 = min(BUDGET, 30)s。
READY="${WORK}/py_ready"
rm -f "${READY}"
# R4 会合次序（重写 R3 的 +10 余量方案——其前提握手本身死锁）：读端
# 先启动（持首个读 fd 等标记，等标记期间不 poll FIFO），写端后启动
# （O_WRONLY|O_NONBLOCK + ENXIO 重试，读端 fd 在场后必配对成功、落
# 标记）。写端预算 = BUDGET + 10：读端先启动（其预算窗口先起算），
# +10s 覆盖双进程启动时延与标记等待，保证写端活过读端的整个预算窗口
# ——否则预算末尾 FIFO 无写者 -> POLLHUP+read==0 被判为变体-1 签名，
# 结构性假阳性 PASS，"预算内未触发"（读端 timeout 124）分支不可达。
WRITER_BUDGET=$(( BUDGET + 10 ))
timeout "${BUDGET}" "${WORK}/old_cpp" "${WORK}/resp.fifo" 100000000 \
    "${READY}" 30 \
    > "${WORK}/cpp.out" 2>&1 &
CPP_PID=$!
timeout "${WRITER_BUDGET}" python3 "${WORK}/old_py.py" "${WORK}/resp.fifo" "${WRITER_BUDGET}" \
    "${READY}" \
    > "${WORK}/py.out" 2>&1 &
PY_PID=$!
wait "${CPP_PID}"
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
if [[ ${CPP_RC} -eq 124 || ${CPP_RC} -eq 4 ]]; then
  echo "[bridge_race_stress_repro] NOTE: 预算内未触发(竞态负载相关;124=timeout/4=迭代上限);" \
     "可加大预算或用 taskset 收紧 CPU 提高触发率" >&2
  exit 1
fi
# 其余退出码 = 读端运行期错误(open/poll 失败或标记超时,见 cpp.out)，
# 不是"预算内未触发"——独立分支点名，避免误导性的加预算建议。
echo "[bridge_race_stress_repro] ERROR: 读端运行期失败(rc=${CPP_RC}," \
     "非竞态签名也非预算耗尽;见上方 cpp/py 输出)" >&2
exit 1
