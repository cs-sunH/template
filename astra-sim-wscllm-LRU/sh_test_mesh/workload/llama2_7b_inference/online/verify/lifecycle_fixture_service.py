#!/usr/bin/env python3
"""lifecycle_fixture_service.py -- 步骤 1-10 IDLE/注入 lifecycle fixture 的决策服务。

run_online_idle_fixture.sh 的 Python 端(合同② 目标 5 五态迁移):
  - 对每个 ARRIVAL 发一个 1ns compute prefill 节点 + watch(stage=prefill,
    generation=0);
  - 对每个 PREFILL_DRAIN 发一个 1ns compute decode 节点 + watch
    (stage=decode, generation=1);
  - DECODE_COMPLETION / REQUEST_COMPLETE 只回空批。

请求经真实物理完成:C++ 侧节点完成 -> WatchRegistry fire -> svc
on_request_completed(五态 IDLE->ACTIVE->IDLE 的返回迁移在 C++ 侧
ServiceCoordinator,步骤 1-10)。本服务无 manifest、无策略——只驱动生命
周期机制;调度逻辑由步骤 1-9 的 strategy E2E 覆盖。

日志:每次 delivery 输出一行 `[fixture] delivery ...`(含 tick / reasons /
arrivals / completed_groups / wall_ms),fixture 脚本据此断言注入的
"指定世界 tick"与完成次序。
"""

import argparse
import json
import os
import sys
import time

_ONLINE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_WORKLOAD_DIR = os.path.dirname(_ONLINE_DIR)
for _path in (_ONLINE_DIR, _WORKLOAD_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from online.decision_bridge import BridgeServer  # noqa: E402

STAGE_PREFILL = "prefill"
STAGE_DECODE = "decode"


class LifecycleHandler:
    """极简决策 handler:arrival -> prefill 节点;prefill drain -> decode 节点。

    节点 id 按 rank 单调递增且跨批次不复用(C++ 的 (rank, json id) -> store
    id 映射是持久化的,见 main_online.cc 的 store_ids)。
    """

    def __init__(self):
        self.next_ids = {}  # rank -> 下一个 json node id
        self.pending_decode = {}  # request_id -> bool(prefill 已发、decode 未发)
        self.delivery_count = 0

    def _next_id(self, rank):
        nid = self.next_ids.get(rank, 1)
        self.next_ids[rank] = nid + 1
        return nid

    @staticmethod
    def _node(rank, nid, request_id, stage, generation):
        return {
            "id": nid,
            "rank": rank,
            "type": 4,  # Compute
            "name": "lifecycle_fixture_{}_{}".format(stage, nid),
            "request_id": request_id,
            "stage": stage,
            "generation": generation,
            "is_cpu_op": False,
            "is_timer_op": False,
            "inputs_values": "",
            "compute": {"num_ops": 1, "tensor_size": 1, "runtime_ns": 1},
            "comm": {"bytes": 0, "src": 0, "dst": 0, "tag": 0},
            "coll": {"comm_type": 0, "bytes": 0, "priority": 0, "pg_name": "", "involved_dim": []},
        }

    def __call__(self, request):
        self.delivery_count += 1
        seq = request.get("delivery_sequence", -1)
        tick = request.get("tick", -1)
        reasons = request.get("reasons", [])
        arrivals = request.get("arrivals", [])
        groups = request.get("completed_groups", [])
        print(
            "[fixture] delivery seq={} tick={} reasons={} arrivals={} "
            "groups={} wall_ms={}".format(
                seq,
                tick,
                json.dumps(reasons),
                json.dumps(arrivals, sort_keys=True),
                json.dumps(groups, sort_keys=True),
                int(time.time() * 1000),
            ),
            flush=True,
        )

        nodes = []
        watches = []
        for arrival in arrivals:
            rid = arrival.get("request_id", "")
            self.pending_decode[rid] = True
            nid = self._next_id(0)
            nodes.append(self._node(0, nid, rid, STAGE_PREFILL, 0))
            watches.append({
                "request_id": rid,
                "stage": STAGE_PREFILL,
                "generation": 0,
                "members": {"0": nid},
                "statuses": ["Success"],
            })
        for group in groups:
            rid = group.get("request_id", "")
            stage = group.get("stage", "")
            if stage == STAGE_PREFILL and self.pending_decode.get(rid):
                self.pending_decode[rid] = False
                nid = self._next_id(0)
                nodes.append(self._node(0, nid, rid, STAGE_DECODE, 1))
                watches.append({
                    "request_id": rid,
                    "stage": STAGE_DECODE,
                    "generation": 1,
                    "members": {"0": nid},
                    "statuses": ["Success"],
                })

        return {
            "nodes": nodes,
            "parent_edges": [],
            "watches": watches,
            "assignments": [],
            "kv_actions": [],
            "future_alarms": [],
            "batch_id": seq,
        }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="wscllm step-1-10 lifecycle fixture decision service")
    parser.add_argument("--bridge-dir", required=True,
                        help="bridge FIFO/request-response 目录(与 C++ 共享)")
    args = parser.parse_args(argv)

    server = BridgeServer(args.bridge_dir)
    result = server.serve_forever(LifecycleHandler())
    if result != 0:
        raise RuntimeError("serve_forever returned {}".format(result))
    print("[fixture] run end: EOF from C++; fixture service exit 0",
          flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # noqa: BLE001 -- CLI 顶层 fail-closed
        print("lifecycle_fixture_service: fatal: {}: {}".format(
            type(exc).__name__, exc),
            file=sys.stderr)
        sys.exit(1)
