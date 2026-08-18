#!/usr/bin/env python3
"""same_tick_milestone_fixture_service.py -- 步骤 1-11(CORE)同 tick milestone
fixture 的决策服务。

run_online_same_tick_milestone.sh 的 Python 端(方案 step 1-11 / 仿真加速
分析.md §4.3 最高优先级专门 fixture):
  - ARRIVAL:发一个**真实同步完成的控制节点**(METADATA_NODE,type=1,正常
    issue 路径;inputs_values 为空 -> issue_pytorch_pg_metadata 直接返回 ->
    skip_invalid 在 issue 的同 tick 内以 Skipped 状态完成)+ watch
    (statuses ["Skipped"])。构造注意:不用零时长 compute 节点——
    Workload::issue_replay 会把 runtime_ns==0 强制为 1(Workload.cc),完成
    tick 不再是 T;
  - PREFILL_DRAIN:发一个 1ns compute decode 节点 + watch
    (statuses ["Success"]);
  - DECODE_COMPLETION / REQUEST_COMPLETE:空批。

四条断言由 runner 脚本承担(方案 step 1-11 a/b/c/d):
  (a) PREFILL_DRAIN 不在 ARRIVAL 的同一 Python 调用内被处理(单 tick 单次
      delivery,Python 不可重入);
  (b) 下一 delivery 的唤醒机制显式存在并记录:seq=1 的 delivery 为
      tick=T+1 且 deferred_from_tick=T(StateDelta 的 T->T+1 延后记录);
  (c) milestone 在下一 delivery epoch 交付,decode 图随后正常提交
      (seq=2 reasons=[DECODE_COMPLETION, REQUEST_COMPLETE]),request
      完成(completed=1,双进程 exit 0);
  (d) 无事件丢失(no_decision_python_callback_count==0)、无死锁(轮询有界)。

日志:每次 delivery 一行 `[fixture] delivery ...`(seq / tick /
deferred_from_tick / reasons / arrivals / completed_groups / wall_ms)。
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


class SameTickMilestoneHandler:
    """ARRIVAL -> 同步完成控制节点(METADATA,type=1);PREFILL_DRAIN -> 1ns decode。"""

    def __init__(self):
        self.next_ids = {}  # rank -> 下一个 json node id
        self.pending_decode = {}  # request_id -> bool(prefill 已发、decode 未发)
        self.delivery_count = 0

    def _next_id(self, rank):
        nid = self.next_ids.get(rank, 1)
        self.next_ids[rank] = nid + 1
        return nid

    @staticmethod
    def _metadata_node(rank, nid, request_id, stage, generation):
        # METADATA_NODE(type=1):真实同步完成控制节点。inputs_values 为空 ->
        # issue_pytorch_pg_metadata 直接返回 -> skip_invalid 在 issue 的同
        # tick 内以 Skipped 完成(正常 issue 路径;不是零时长 compute 的
        # 强制 runtime=1 路径——方案 step 1-11 fixture 构造注意)。
        return {
            "id": nid,
            "rank": rank,
            "type": 1,  # Metadata(ChakraNodeType::METADATA_NODE)
            "name": "stm_control_{}_{}".format(stage, nid),
            "request_id": request_id,
            "stage": stage,
            "generation": generation,
            "is_cpu_op": False,
            "is_timer_op": False,
            "inputs_values": "",
            "compute": {"num_ops": 0, "tensor_size": 0, "runtime_ns": 0},
            "comm": {},
            "coll": {},
        }

    @staticmethod
    def _compute_node(rank, nid, request_id, stage, generation):
        return {
            "id": nid,
            "rank": rank,
            "type": 4,  # Compute
            "name": "stm_decode_{}_{}".format(stage, nid),
            "request_id": request_id,
            "stage": stage,
            "generation": generation,
            "is_cpu_op": False,
            "is_timer_op": False,
            "inputs_values": "",
            "compute": {"num_ops": 1, "tensor_size": 1, "runtime_ns": 1},
            "comm": {},
            "coll": {},
        }

    def __call__(self, request):
        self.delivery_count += 1
        seq = request.get("delivery_sequence", -1)
        tick = request.get("tick", -1)
        deferred = request.get("deferred_from_tick", 0)
        reasons = request.get("reasons", [])
        arrivals = request.get("arrivals", [])
        groups = request.get("completed_groups", [])
        print(
            "[fixture] delivery seq={} tick={} deferred_from_tick={} "
            "reasons={} arrivals={} groups={} wall_ms={}".format(
                seq,
                tick,
                deferred,
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
            nodes.append(
                self._metadata_node(0, nid, rid, STAGE_PREFILL, 0))
            watches.append({
                "request_id": rid,
                "stage": STAGE_PREFILL,
                "generation": 0,
                "members": {"0": nid},
                "statuses": ["Skipped"],
            })
        for group in groups:
            rid = group.get("request_id", "")
            stage = group.get("stage", "")
            if stage == STAGE_PREFILL and self.pending_decode.get(rid):
                self.pending_decode[rid] = False
                nid = self._next_id(0)
                nodes.append(
                    self._compute_node(0, nid, rid, STAGE_DECODE, 1))
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
        description="sh_1.0 step-1-11 same-tick milestone fixture "
                    "decision service")
    parser.add_argument("--bridge-dir", required=True,
                        help="bridge FIFO/request-response 目录(与 C++ 共享)")
    args = parser.parse_args(argv)

    server = BridgeServer(args.bridge_dir)
    result = server.serve_forever(SameTickMilestoneHandler())
    if result != 0:
        raise RuntimeError("serve_forever returned {}".format(result))
    print("[fixture] run end: EOF from C++; fixture service exit 0",
          flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # noqa: BLE001 -- CLI 顶层 fail-closed
        print("same_tick_milestone_fixture_service: fatal: {}: {}".format(
            type(exc).__name__, exc),
            file=sys.stderr)
        sys.exit(1)
