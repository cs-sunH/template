#!/usr/bin/env python3
"""graph_batch_audit.py -- sh_1.0 阶段 3(方案 §8)GraphBatch 验收对平审计。

三项验收的对平证据(材料 = 官方 runner 运行目录):
  1. graph_batch == delivery 对平:bridge/request_*.json 的 delivery_sequence
     集合 == {0..N-1}(无跳号/无重复);results/graph_batch_digests.jsonl
     每交付恰一批(batch_id == delivery_sequence 由 C++ validate 规则 B 与
     Python ack 门双侧强制,此处复核数量与覆盖);
  2. 正式路径 single_node_bridge_count == 0(cpp.log phase-5 计数器行);
  3. 总账:graph_batch_count == N,digest 行 node_count 合计 ==
     计数器 total_nodes;Python summary delivery_count == ack_count == N。

用法:
  python3 online/verify/graph_batch_audit.py <run_dir> [<run_dir> ...]

退出码:全部对平 -> 0;任何不平 -> 1(fail-closed,不放宽)。
"""

import glob
import json
import os
import re
import sys


def audit(run_dir):
    out = {"run_dir": run_dir}
    requests = []
    for path in glob.glob(os.path.join(run_dir, "bridge", "request_*.json")):
        requests.append(json.load(open(path, encoding="utf-8")))
    requests.sort(key=lambda d: d["delivery_sequence"])
    seqs = [d["delivery_sequence"] for d in requests]
    out["deliveries"] = len(requests)
    assert seqs == list(range(len(requests))), (
        "delivery sequences are not gap-free 0..N-1",
        seqs[:5], seqs[-5:] if len(seqs) > 5 else seqs)

    digests = []
    with open(os.path.join(run_dir, "results",
                           "graph_batch_digests.jsonl"), encoding="utf-8") as f:
        for line in f:
            if line.strip():
                digests.append(json.loads(line))
    out["digest_rows"] = len(digests)
    assert len(digests) == len(requests), (len(digests), len(requests))
    assert [d["delivery_sequence"] for d in digests] == seqs, \
        "digest rows do not cover exactly the delivery sequence set"
    out["digest_node_sum"] = sum(d["node_count"] for d in digests)

    with open(os.path.join(run_dir, "cpp.log"), encoding="utf-8") as f:
        cpp_log = f.read()
    m = re.search(r"phase-5 commit counters: graph_batch_count=(\d+) "
                  r"single_node_bridge_count=(\d+) total_nodes=(\d+) "
                  r"avg_nodes_per_batch=(\S+) max_nodes_per_batch=(\d+) "
                  r"total_watches=(\d+)", cpp_log)
    assert m, "phase-5 commit counters line missing from cpp.log"
    graph_batch_count = int(m.group(1))
    single_node_bridge_count = int(m.group(2))
    total_nodes = int(m.group(3))
    out["cpp_graph_batch_count"] = graph_batch_count
    out["cpp_single_node_bridge_count"] = single_node_bridge_count
    out["cpp_total_nodes"] = total_nodes
    out["cpp_avg_nodes_per_batch"] = m.group(4)
    out["cpp_max_nodes_per_batch"] = int(m.group(5))
    out["cpp_total_watches"] = int(m.group(6))
    # 门槛 2:正式路径 single_node_bridge_count == 0(方案 §8.3)。
    assert single_node_bridge_count == 0, single_node_bridge_count
    # 门槛 1/3:数量与覆盖对平。
    assert graph_batch_count == len(requests), \
        (graph_batch_count, len(requests))
    assert total_nodes == out["digest_node_sum"], \
        (total_nodes, out["digest_node_sum"])

    summary = None
    with open(os.path.join(run_dir, "results",
                           "online_stats.jsonl"), encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                if row.get("summary"):
                    summary = row
    assert summary is not None, "online_stats.jsonl summary row missing"
    out["python_delivery_count"] = summary["delivery_count"]
    out["python_ack_count"] = summary["ack_count"]
    assert summary["delivery_count"] == len(requests)
    assert summary["ack_count"] == summary["delivery_count"]
    out["verdict"] = "PASS"
    return out


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    for run_dir in sys.argv[1:]:
        result = audit(run_dir)
        print("[graph_batch_audit] %s: %s (deliveries=%d, nodes=%d, "
              "avg/batch=%s, max/batch=%d, single_node_bridge=%d)"
              % (result["run_dir"], result["verdict"], result["deliveries"],
                 result["cpp_total_nodes"], result["cpp_avg_nodes_per_batch"],
                 result["cpp_max_nodes_per_batch"],
                 result["cpp_single_node_bridge_count"]))
    print("ALL RUNS PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
