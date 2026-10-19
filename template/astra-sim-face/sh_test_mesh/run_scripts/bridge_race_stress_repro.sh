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
# 语义声明(2026-09-26 审查核查):本脚本是独立的竞态复现演示工具,不是
# 门禁/健康自检——它不构建也不运行当前仓任何代码,只在 /tmp 复刻"旧协议
# 副本"。因此 PASS(退出 0)= 旧竞态在复刻副本中按预期复现(演示成功),
# exit 1 = 预算内未复现(负载相关,不代表当前仓有或没有缺陷)。任何脚本
# 不得把本脚本退出码当作当前仓代码的门禁判据;README 中的"机制层回归
# fixtures"条目应据此口径理解。
#
# 判定修正(2026-09-29):原版写端预算比读端早 0.2s 耗尽——预算末尾写端
# 进程退出产生的"真实 EOF"恰好落进读端仍在 pending 的 poll(POLLHUP+
# read==0),被误判成变体 1,"预算内未复现"分支不可达、脚本恒 exit 0。
# 现双保险:(1) 写端预算 = 读端预算 + GRACE 秒,读端超时发生时写端必然
# 仍存活;(2) 读端在 read==0 时以 kill(写端pid,0) 验活——写端进程存活
# 才判变体 1(rc 3),已退出则为真实 EOF(rc 5,运行无效,不算复现)。
# 三种时序推演:正常竞态→rc 3 走 PASS;读端先超时→rc 124 走"未复现";
# 写端被外部先杀→rc 5 走 INVALID,不再产生假 PASS。
#
# 用法: bash bridge_race_stress_repro.sh [预算秒数,缺省 60]
set -uo pipefail

BUDGET=${1:-60}
case ${BUDGET} in
  ''|*[!0-9]*) echo "usage: bridge_race_stress_repro.sh [预算秒数(正整数)]" >&2; exit 2 ;;
esac
WORK=$(mktemp -d /tmp/bridge_race_repro.XXXXXX)
trap 'rm -rf "${WORK}"' EXIT

# ---- 复刻旧 C++ 模式(逐交换 open/poll/read/close)----
cat > "${WORK}/old_cpp.c" <<'EOF'
#include <errno.h>
#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
static const char* F;
int main(int argc, char** argv) {
    if (argc < 4) { fprintf(stderr, "usage: %s <fifo> <iters> <writer_pid>\n", argv[0]); return 2; }
    F = argv[1];
    long budget_iters = atol(argv[2]);
    pid_t writer_pid = (pid_t)atol(argv[3]);
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
            /* read()==0 有两种成因,必须区分(2026-09-29 判定修正):
               - 写端进程仍存活:旧协议"写端 fd 1->0 关闭转移 + 空缓冲"
                 被 poll 看成 POLLHUP => 变体 1 假 EOF(要复现的签名);
               - 写端进程已退出:真实 EOF,不是竞态,不得计入复现。 */
            if (writer_pid > 0 && kill(writer_pid, 0) == 0) {
                fprintf(stderr,
                        "[repro] FALSE-CRASH at iter %ld: poll revents=%u "
                        "(POLLHUP=%u) read==0 while writer alive\n",
                        i, pfd.revents, (unsigned)(pfd.revents & POLLHUP) == POLLHUP);
                printf("[repro] REPRODUCED false 'Python side crashed' at iter %ld\n", i);
                return 3;  /* the F1 signature */
            }
            fprintf(stderr,
                    "[repro] real EOF at iter %ld: writer process %ld gone "
                    "(not the race)\n", i, (long)writer_pid);
            return 5;
        }
        if (n < 0) { perror("read"); return 2; }
    }
    printf("[repro] no false-crash within budget iters\n");
    return 4;
}
EOF

# ---- 复刻旧 Python 模式(逐交换 open/write/close;EPIPE 容错继续,
#      让互补变体不提前终结压测)。open 用 O_NONBLOCK+ENXIO 重试而非阻塞
#      open(2026-09-29 判定修正):写端必须自限于预算(读端超时路径要求
#      写端仍存活、不得挂住收尾 wait),且 $! 须是真 python 进程供读端
#      kill(pid,0) 验活;逐交换 open/close 的写端计数 1->0 转移(变体 1
#      的触发面)不变。----
cat > "${WORK}/old_py.py" <<'EOF'
import errno, os, sys, time
F = sys.argv[1]
deadline = time.time() + float(sys.argv[2])
epipes = 0
while time.time() < deadline:
    try:
        fd = os.open(F, os.O_WRONLY | os.O_NONBLOCK)
    except OSError as e:
        if e.errno == errno.ENXIO:  # no reader present; retry within deadline
            time.sleep(0.0005)
            continue
        raise
    try:
        os.write(fd, b"\n")
    except BrokenPipeError:
        epipes += 1  # variant-2 of the same race family (observed, logged)
    finally:
        os.close(fd)
print("[repro] python writer done; broken_pipes(observed variant 2) =", epipes)
EOF

gcc -O2 -o "${WORK}/old_cpp" "${WORK}/old_cpp.c" || { echo "gcc failed" >&2; exit 2; }
mkfifo "${WORK}/resp.fifo"

# 读端就绪后再启动写端(第一个交换的配对)。写端预算 = 读端预算 + GRACE:
# 读端 timeout(124) 发生时写端进程必然仍存活,预算末尾不再存在"写端先退出
# 的真实 EOF"窗口(修正前为 0.2s 反向窗口,恒被误判成复现)。
GRACE=5
python3 "${WORK}/old_py.py" "${WORK}/resp.fifo" "$((BUDGET + GRACE))" \
    > "${WORK}/py.out" 2>&1 &
PY_PID=$!
sleep 0.2

timeout "${BUDGET}" "${WORK}/old_cpp" "${WORK}/resp.fifo" 100000000 "${PY_PID}" \
    > "${WORK}/cpp.out" 2>&1
CPP_RC=$?
# 判定已完成,写端无观测价值:提前收尾,避免收尾 wait 滞留最多 GRACE 秒。
kill "${PY_PID}" 2>/dev/null
wait "${PY_PID}" 2>/dev/null

echo "--- cpp:"; cat "${WORK}/cpp.out"
echo "--- py:"; cat "${WORK}/py.out"

if [[ ${CPP_RC} -eq 3 ]]; then
  echo "[bridge_race_stress_repro] PASS: 旧协议竞态(假 EOF 误杀)在本机复现;" \
       "正式代码已改双侧长连接消除该窗口"
  exit 0
elif [[ ${CPP_RC} -eq 5 ]]; then
  echo "[bridge_race_stress_repro] INVALID: 读端观察到写端进程已退出的真实 EOF" \
       "(非竞态,不算复现;本脚本时序下写端预算比读端长 ${GRACE}s,该路径只应" \
       "由外部 kill/OOM 触发)" >&2
  exit 1
elif [[ ${CPP_RC} -eq 124 ]]; then
  echo "[bridge_race_stress_repro] NOTE: 读端预算 ${BUDGET}s 耗尽未触发(竞态负载相关);" \
       "可加大预算或用 taskset 收紧 CPU 提高触发率" >&2
  exit 1
fi
echo "[bridge_race_stress_repro] NOTE: 读端异常退出(rc=${CPP_RC},预期 3/5/124);" \
     "按未复现处理" >&2
exit 1
