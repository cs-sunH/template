#!/usr/bin/env python3
"""wakeup_guard_fixture_service.py -- 缺陷 C 回归 fixture 的决策服务
(2026-08-16, synced from face-defectfix2-done; 源分析:
face主动测试错误分析.md 缺陷 C)。

run_online_wakeup_guard_fixture.sh 的 Python 端,两个场景:

场景 1(f7-blueprint,F7 末态蓝图的健康形态):
  - ARRIVAL(r0):同步完成控制节点(METADATA,Skipped watch)+ 一个
    【未到期 future alarm】(r1 @ T+1000)——最后一笔交付携带未到期
    future alarm 的蓝图要素;
  - 里程碑与 r1 的 ARRIVAL 在 T+1000 的同一 epoch 交付(主队列一度只
    剩该 alarm);
  - r1 照常准入(metadata+decode)→ 两请求全部完成,CloseInput 后
    正常结束(exit 0)——证明修复后的空队列分支不会在健康蓝图形态上
    误触发 fail-closed。

场景 2(defer-dead-end,唤醒真空死端):
  - ARRIVAL(r0):返回空批(0 节点/0 watch/0 alarm)= 策略 defer;
  - C++ 侧 active=1、队列空、mailbox 空;
  - runner 注入 CloseInput 后:旧代码 busy-spin(无进展、永不退出),
    修复后立即 fail-closed abort 并打印 "lost-wakeup dead end" 诊断
    (active=1 计数入消息)。

日志:每次 delivery 一行 [fixture] delivery(seq/tick/deferred/
reasons/arrivals/groups)。
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


class WakeupGuardHandler:
    """scenario=f7-blueprint | defer-dead-end。"""

    def __init__(self, scenario):
        self.scenario = scenario
        self.next_id = 1
        self.prefill_sent = {}   # request_id -> bool(decode 未发)
        self.delivery_count = 0

    def _nid(self):
        nid = self.next_id
        self.next_id += 1
        return nid

    @staticmethod
    def _metadata_node(nid, request_id):
        return {
            "id": nid, "rank": 0, "type": 1,
            "name": "wg_control_{}".format(nid),
            "request_id": request_id, "stage": STAGE_PREFILL,
            "generation": 0,
            "is_cpu_op": False, "is_timer_op": False,
            "inputs_values": "",
            "compute": {"num_ops": 0, "tensor_size": 0, "runtime_ns": 0},
            "comm": {}, "coll": {},
        }

    @staticmethod
    def _compute_node(nid, request_id):
        return {
            "id": nid, "rank": 0, "type": 4,
            "name": "wg_decode_{}".format(nid),
            "request_id": request_id, "stage": STAGE_DECODE,
            "generation": 1,
            "is_cpu_op": False, "is_timer_op": False,
            "inputs_values": "",
            "compute": {"num_ops": 1, "tensor_size": 1, "runtime_ns": 1},
            "comm": {}, "coll": {},
        }

    def __call__(self, request):
        self.delivery_count += 1
        seq = request.get("delivery_sequence", -1)
        tick = request.get("tick", -1)
        reasons = request.get("reasons", [])
        arrivals = request.get("arrivals", [])
        groups = request.get("completed_groups", [])
        print("[fixture] delivery seq={} tick={} deferred_from_tick={} "
              "reasons={} arrivals={} groups={}".format(
                  seq, tick, request.get("deferred_from_tick", 0),
                  json.dumps(reasons),
                  json.dumps([a.get("request_id") for a in arrivals]),
                  json.dumps([(g.get("request_id"), g.get("stage"))
                              for g in groups])),
              flush=True)

        if self.scenario == "defer-dead-end":
            # 空批 defer:0 节点/0 watch/0 future alarm。C++ 侧 active 保持
            # 1,主队列/mailbox 双空 —— 唤醒真空死端(由 runner 触发 CloseInput
            # 后观察修复前 spin / 修复后 fail-closed)。
            return {
                "nodes": [], "parent_edges": [], "watches": [],
                "assignments": [], "kv_actions": [], "future_alarms": [],
                "batch_id": seq,
            }

        # ---- f7-blueprint ----
        nodes = []
        watches = []
        future_alarms = []
        for arrival in arrivals:
            rid = arrival.get("request_id", "")
            self.prefill_sent[rid] = True
            nid = self._nid()
            nodes.append(self._metadata_node(nid, rid))
            watches.append({
                "request_id": rid, "stage": STAGE_PREFILL,
                "generation": 0, "members": {"0": nid},
                "statuses": ["Skipped"],
            })
            if rid.endswith("_r0"):
                # F7 蓝图要素:r0 的 ARRIVAL 批携带未到期 future alarm
                # (r1 @ tick+1000;schema: GraphBatchCommitter 校验口径)。
                future_alarms.append({
                    "arrival_world_ns": tick + 1000,
                    "envelope": {
                        "request_id": rid + "1",
                        "session_id": arrival.get("session_id", "wg_s0"),
                        "turn_index": 1,
                        "prefill_length": 128,
                        "decode_length": 8,
                        "inter_request_interval_ns": 0,
                    },
                })
                print("[fixture] future_alarm scheduled: {} @ tick+1000"
                      .format(rid + "1"), flush=True)
        for group in groups:
            rid = group.get("request_id", "")
            stage = group.get("stage", "")
            if stage == STAGE_PREFILL and self.prefill_sent.get(rid):
                self.prefill_sent[rid] = False
                nid = self._nid()
                nodes.append(self._compute_node(nid, rid))
                watches.append({
                    "request_id": rid, "stage": STAGE_DECODE,
                    "generation": 1, "members": {"0": nid},
                    "statuses": ["Success"],
                })
        return {
            "nodes": nodes, "parent_edges": [], "watches": watches,
            "assignments": [], "kv_actions": [],
            "future_alarms": future_alarms,
            "batch_id": seq,
        }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="defect-C wakeup guard fixture decision service")
    parser.add_argument("--bridge-dir", required=True)
    parser.add_argument("--scenario", required=True,
                        choices=("f7-blueprint", "defer-dead-end"))
    args = parser.parse_args(argv)

    server = BridgeServer(args.bridge_dir)
    handler = WakeupGuardHandler(args.scenario)
    result = server.serve_forever(handler)
    if result != 0:
        raise RuntimeError("serve_forever returned {}".format(result))
    print("[fixture] run end: EOF from C++ (deliveries={})".format(
        handler.delivery_count), flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # noqa: BLE001 -- CLI 顶层 fail-closed
        print("wakeup_guard_fixture_service: fatal: {}: {}".format(
            type(exc).__name__, exc), file=sys.stderr)
        sys.exit(1)
