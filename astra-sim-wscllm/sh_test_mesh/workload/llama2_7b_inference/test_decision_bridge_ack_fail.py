#!/usr/bin/env python3
"""test_decision_bridge_ack_fail.py -- P2-18 钉死测试（桥级）。

契约（2026-09-28 修复）：commit_ack 处理失败时，结构化 error 必须落到
**C++ 下一步必然读取的 response 槽位 ack_seq+1**（C++ deliver_and_receive
消费序：wait_response_byte(K+1) → read response_<K+1> → error 非空即
bridge_fatal），而不是写回已消费并 unlink 的 response_<ack_seq>——旧实现
写到 seq K 的 response 永不可达，注入的 notify 字节反被 C++ 当下一笔交付
(K+1) 的就绪信号，以指向错误序号的 "response missing or corrupt" abort，
真实错误丢失。

本测试用真实 FIFO 通道跑一笔完整交付（request→response→ack），在
on_commit_ack 注入异常后，以 C++ 的消费序验证 error 可达：
  - response_1.json 存在、error 携带真实异常文本、
    source_delivery_sequence == 1（槽位路由）；
  - 不再生成的 response_0.json（旧实现错写槽位）；
  - Python 侧 fail-closed 非 0 退出、stderr 留痕。
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

MODULE_DIR = Path(__file__).resolve().parent
ONLINE_DIR = MODULE_DIR / "online"

_CHILD = r"""
import os, sys
bridge_dir, online_dir = sys.argv[1], sys.argv[2]
sys.path.insert(0, online_dir)
from decision_bridge import BridgeServer

def bad_bookkeeping(ack):
    raise RuntimeError("ack bookkeeping exploded")

def handler(request):
    return {"source_delivery_sequence": request["delivery_sequence"]}

BridgeServer(bridge_dir).serve_forever(handler, on_commit_ack=bad_bookkeeping)
"""


def _write_json(path: Path, payload: dict) -> None:
    tmp = str(path) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(payload))
    os.replace(tmp, str(path))


class AckFailRoutingTests(unittest.TestCase):
    """P2-18: ack 失败 → error 落下一 response 槽位（K+1），可被 C++ 消费。"""

    # (2026-10-01, I2 复审) FIFO 是桥协议的 POSIX 语义;win32 缺 os.mkfifo,
    # 收集/运行直接 AttributeError,按平台显式跳过(官方目标环境为 Linux/WSL)。
    @unittest.skipIf(os.name == "nt", "bridge FIFOs are POSIX-only")
    def test_ack_failure_error_reaches_next_response_slot(self):
        bridge_dir = Path(tempfile.mkdtemp(prefix="bridge_ackfail_"))
        # (2026-10-01, D2 复审) Temp-dir hygiene: register cleanup so the
        # bridge directory (FIFOs + request/response JSONs) never leaks,
        # matching the repo's atexit/addCleanup standard (synthetic.py /
        # test_hbm_watermark.py).
        self.addCleanup(shutil.rmtree, bridge_dir, True)
        req_fifo = bridge_dir / "req_notify.fifo"
        resp_fifo = bridge_dir / "resp_notify.fifo"
        os.mkfifo(req_fifo)
        os.mkfifo(resp_fifo)

        proc = subprocess.Popen(
            [sys.executable, "-u", "-c", _CHILD, str(bridge_dir),
             str(ONLINE_DIR)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        try:
            # 桥打开顺序：先 req 读端（阻塞至本侧写端）再 resp 写端（阻塞
            # 至本侧读端）。本侧先开 req 写端，再以非阻塞开 resp 读端。
            req_fd = os.open(str(req_fifo), os.O_WRONLY)
            resp_fd = os.open(str(resp_fifo), os.O_RDONLY | os.O_NONBLOCK)
            os.set_blocking(resp_fd, True)

            def notify():
                os.write(req_fd, b"\n")

            def wait_response_byte():
                data = os.read(resp_fd, 1)
                if not data:
                    self.fail("resp FIFO EOF（Python 侧提前退出）")

            # -- 交付 0：request → response → ack（C++ 消费序）-----------
            _write_json(bridge_dir / "request_0.json",
                        {"schema_version": 1, "delivery_sequence": 0})
            notify()
            wait_response_byte()
            response_0 = json.loads(
                (bridge_dir / "response_0.json").read_text(encoding="utf-8"))
            self.assertEqual(response_0["source_delivery_sequence"], 0)
            self.assertEqual(response_0.get("error", ""), "")
            os.unlink(bridge_dir / "response_0.json")  # C++ 消费后 unlink
            _write_json(bridge_dir / "commit_ack_0.json",
                        {"schema_version": 1, "delivery_sequence": 0,
                         "batch_id": 0, "success": True})
            notify()  # C++ send_commit_ack 的 notify_python()

            # -- ack_0 处理失败 → 修复后 error 落 response_1.json ---------
            # 桥在 ack 循环里 _fail_after_ack(0)：写 response_1 + notify
            # 字节 + exit(1)。本侧按 C++ 交付 1 的消费序读取。
            wait_response_byte()  # 吸收 fail 注入的就绪字节（旧缺陷：错序）
            slot_1 = bridge_dir / "response_1.json"
            self.assertTrue(slot_1.is_file(),
                            "error 未落到下一 response 槽位 K+1（不可达）")
            routed = json.loads(slot_1.read_text(encoding="utf-8"))
            self.assertEqual(routed["source_delivery_sequence"], 1)
            # on_commit_ack 抛非桥异常 → 通用臂格式 "Type: message"。
            self.assertIn("RuntimeError: ack bookkeeping exploded",
                          routed["error"])
            # 旧实现把 error 写回已消费的槽位 0（永不可达）——不得再现。
            self.assertFalse((bridge_dir / "response_0.json").is_file(),
                             "ack 失败错误被写回已消费槽位 0（永不可达）")

            os.close(req_fd)
            os.close(resp_fd)
        finally:
            # 正常收尾：桥在 _fail_after_ack 里 sys.exit(1)，等它退出
          # （超时才 kill）；先 wait 后读 stderr（子进程已 EOF）。
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            stderr = proc.stderr.read().decode("utf-8", "replace")

        # fail-closed：Python 侧非 0 退出 + stderr 留痕。
        self.assertEqual(proc.returncode, 1, stderr)
        self.assertIn("fail-closed at ack seq 0", stderr)
        self.assertIn("ack bookkeeping exploded", stderr)


if __name__ == "__main__":
    unittest.main()
