#!/usr/bin/env python3
"""bridge_echo.py -- 步骤 1-7 回环 fixture 的 Python echo 服务。

用法:
    python3 bridge_echo.py <bridge_dir> <ack_receipt>

handler 固定返回一组满足冻结 inner schema 的 nodes/parent_edges/
assignments/kv_actions 数组供 C++ 侧 typed 断言;schema_version/batch_id/
source_delivery_sequence 按协议补全。C1(2026-08-29): C++ 侧响应改为
parse_graph_batch 单次结构化解析,未知顶层键 fail-closed,故回显里不再
携带测试专用的 echo_of_request 键 —— 请求侧往返无损断言由 C++ fixture
直接对 request_<seq>.json(原子发布后)与 build_request_json 输出比对,
不再借道响应回显。commit ack 到达时把 ack dict 追加为 ack_receipt 的
一行(逐行 JSON),C++ 侧据此断言 ack 通知到达且幂等(重复 seq 只落一行)。
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from decision_bridge import BridgeServer  # noqa: E402


def _echo_node(request_id, node_id, name):
    return {
        "id": node_id,
        "rank": 0,
        "type": 4,
        "name": name,
        "request_id": request_id,
        "stage": "prefill",
        "generation": 0,
        "is_cpu_op": False,
        "is_timer_op": False,
        "inputs_values": "",
        "compute": {"num_ops": 1, "tensor_size": 1, "runtime_ns": 1},
        "comm": {"bytes": 0, "src": 0, "dst": 0, "tag": 0},
        "coll": {"comm_type": 0, "bytes": 0, "priority": 0, "pg_name": "",
                 "involved_dim": []},
    }


def main():
    bridge_dir = sys.argv[1]
    ack_receipt = sys.argv[2]

    def echo(request):
        arrivals = request.get("arrivals") or [{}]
        request_id = arrivals[0].get("request_id", "s0_r0")
        return {
            "batch_id": request["delivery_sequence"],
            "source_delivery_sequence": request["delivery_sequence"],
            "nodes": [_echo_node(request_id, 0, "echo_node_1"),
                      _echo_node(request_id, 1, "echo_node_2")],
            "parent_edges": [{"rank": 0, "kind": "data", "from": 0, "to": 1}],
            "watches": [],
            "assignments": [{"request_id": request_id,
                             "prefill_instance_index": 0,
                             "decode_instance_index": 0}],
            "kv_actions": [{"event_type": "echo",
                            "trigger_request_id": request_id}],
            "future_alarms": [],
        }

    def on_ack(ack):
        with open(ack_receipt, "a", encoding="utf-8") as target:
            target.write(json.dumps(ack, sort_keys=True) + "\n")

    server = BridgeServer(bridge_dir)
    return server.serve_forever(echo, on_commit_ack=on_ack)


if __name__ == "__main__":
    sys.exit(main())
