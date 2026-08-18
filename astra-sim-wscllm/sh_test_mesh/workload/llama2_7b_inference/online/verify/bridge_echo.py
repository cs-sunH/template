#!/usr/bin/env python3
"""bridge_echo.py -- 步骤 1-7 回环 fixture 的 Python echo 服务。

用法:
    python3 bridge_echo.py <bridge_dir> <ack_receipt>

handler 把完整 request dict 原样回显到 response["echo_of_request"] 里
(字段往返无损断言素材),固定返回一组 nodes/parent_edges/watches/
assignments/kv_actions 数组供 C++ 侧断言;schema_version/batch_id/
source_delivery_sequence 按协议补全。commit ack 到达时把 ack dict 追加为
ack_receipt 的一行(逐行 JSON),C++ 侧据此断言 ack 通知到达且幂等
(重复 seq 只落一行)。
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from decision_bridge import BridgeServer  # noqa: E402


def main():
    bridge_dir = sys.argv[1]
    ack_receipt = sys.argv[2]

    def echo(request):
        return {
            "batch_id": request["delivery_sequence"],
            "source_delivery_sequence": request["delivery_sequence"],
            "nodes": [{"echo_node": 1}, {"echo_node": 2}],
            "parent_edges": [{"from": 0, "to": 1}],
            "watches": [],
            "assignments": [{"echo": "assignment"}],
            "kv_actions": [{"echo": "kv_action"}],
            "echo_of_request": request,  # 测试专用: 往返无损断言素材
        }

    def on_ack(ack):
        with open(ack_receipt, "a", encoding="utf-8") as target:
            target.write(json.dumps(ack, sort_keys=True) + "\n")

    server = BridgeServer(bridge_dir)
    return server.serve_forever(echo, on_commit_ack=on_ack)


if __name__ == "__main__":
    sys.exit(main())
