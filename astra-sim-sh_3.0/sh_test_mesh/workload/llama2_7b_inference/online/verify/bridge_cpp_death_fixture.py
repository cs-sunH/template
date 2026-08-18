#!/usr/bin/env python3
"""bridge_cpp_death_fixture.py -- 缺陷 B 修复回归(2026-08-16, Python 侧;
synced from face-defectfix2-done)。

C++ 进程运行中途死亡 => Python 侧 resp_notify 长连接写端 BrokenPipe =>
BridgePipeError fail-closed 退出 + stderr 留痕(修复 F1 现场 python.log
0 字节、无可诊断性的问题)。

结构:本进程 fork 出"假 C++"子进程,复刻 FileDecisionBridge 的 C++ 侧
fd 语义(与真实现相同的通道操作序列):
  1. req 写端 O_WRONLY|O_NONBLOCK 重试至真实侧读端就绪(同 open_notify);
  2. resp 读端 O_RDONLY|O_NONBLOCK 常开(同 open_notify 的缺陷 B 修复);
  3. 写 request_1.json + 门铃、request_2.json + 门铃;
  4. 保活 ~1.5s(让真实侧完成第 1 笔交付的字节写);
  5. 关闭两个 fd 并 _exit(0) = 模拟 C++ 进程中途死亡。

父进程 = 真实 BridgeServer + echo handler。期望:处理第 2 笔交付时 resp
写端 BrokenPipe => serve_forever 的 BridgePipeError 分支打印
"C++ side is gone" 到 stderr 并 sys.exit(1)。外层 runner 断言退出码 1
且 stderr 含该字样。

用法(由 run_bridge_cpp_death_fixture.sh 调用):
    python3 bridge_cpp_death_fixture.py <bridge_dir>
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from decision_bridge import BridgeServer  # noqa: E402

SCHEMA_VERSION = 1


def _write_json_atomic(path, payload):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as target:
        json.dump(payload, target)
    os.replace(tmp, path)


def _request(seq):
    return {
        "schema_version": SCHEMA_VERSION,
        "delivery_sequence": seq,
        "delivery_epoch": seq,
        "tick": 1000 + seq,
        "deferred_from_tick": 0,
        "reasons": ["ARRIVAL"],
        "arrivals": [{
            "request_id": "s0_r%d" % seq,
            "session_id": "s0",
            "turn_index": 0,
            "prefill_length": 16,
            "decode_length": 4,
            "inter_request_interval_ns": 0,
            "arrival_world_ns": 1000 + seq,
            "ingress_seq": seq,
            "queue_index": seq,
        }],
        "completed_groups": [],
        "completed_nodes": [],
        "retry_items": [],
        "affected_ranks": [],
        "snapshot_handle": {"epoch": seq, "tick": 1000 + seq, "kind": ""},
        "snapshot": {},
        "ledger_summary": {"injected_unfinished": []},
    }


def fake_cpp_child(bridge_dir):
    """复刻 FileDecisionBridge 的 C++ 侧 fd 语义,然后中途死亡。"""
    req_fifo = os.path.join(bridge_dir, "req_notify.fifo")
    resp_fifo = os.path.join(bridge_dir, "resp_notify.fifo")
    wfd = -1
    while wfd < 0:
        try:
            wfd = os.open(req_fifo, os.O_WRONLY | os.O_NONBLOCK)
        except OSError as exc:
            if exc.errno != 6:  # ENXIO: no reader yet
                raise
            time.sleep(0.02)
    rfd = os.open(resp_fifo, os.O_RDONLY | os.O_NONBLOCK)
    # 交换 1: request 文件 + 门铃;等真实侧写出 response_1.json 并完成
    # 字节写(留 0.3s 余量),模拟一次完整交付。
    _write_json_atomic(
        os.path.join(bridge_dir, "request_1.json"), _request(1))
    os.write(wfd, b"\n")
    resp1 = os.path.join(bridge_dir, "response_1.json")
    deadline = time.time() + 10.0
    while not os.path.exists(resp1) and time.time() < deadline:
        time.sleep(0.02)
    time.sleep(0.3)
    # 交换 2: request 文件 + 门铃后【立即】死亡 —— 真实侧的门铃 2 处理
    # (读文件/写 response/写通知字节)至少要一次文件 I/O,死亡必先于
    # 其通知字节落地 => BrokenPipe。
    _write_json_atomic(
        os.path.join(bridge_dir, "request_2.json"), _request(2))
    os.write(wfd, b"\n")
    os.close(wfd)
    os.close(rfd)
    os._exit(0)


def main():
    bridge_dir = sys.argv[1]
    os.makedirs(bridge_dir, exist_ok=True)
    for name in ("req_notify.fifo", "resp_notify.fifo"):
        path = os.path.join(bridge_dir, name)
        if not os.path.exists(path):
            os.mkfifo(path, 0o600)

    pid = os.fork()
    if pid == 0:
        fake_cpp_child(bridge_dir)

    server = BridgeServer(bridge_dir)

    def echo(request):
        return {
            "batch_id": request["delivery_sequence"],
            "source_delivery_sequence": request["delivery_sequence"],
            "nodes": [],
            "parent_edges": [],
            "watches": [],
            "assignments": [],
            "kv_actions": [],
        }

    # 目标路径:第 2 笔交付的 resp 写 BrokenPipe => BridgePipeError 分支
    # 打印 "C++ side is gone" 并 sys.exit(1)。
    server.serve_forever(echo)
    # 走到这里说明 req EOF 先于 BrokenPipe 到达(时序边界),不是本
    # fixture 的目标路径;以非 0 退出暴露给 runner。
    print("bridge_cpp_death_fixture: UNEXPECTED EOF path (no BrokenPipe)",
          file=sys.stderr)
    return 3


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001 -- 顶层 fail-closed
        print("bridge_cpp_death_fixture: fatal: {}: {}".format(
            type(exc).__name__, exc), file=sys.stderr)
        sys.exit(1)
