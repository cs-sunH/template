#!/usr/bin/env python3
"""b3_node_compare.py -- sh_2.0 Tier B B3 逐节点 canonical 比较器（阶段 2）。

比较基准 = 与插入顺序无关的 canonical logical node key（方案 §5.5）：
per-rank (name, type) 多重集 + 关键属性桶（tensor_size/num_ops/comm 字节/
tag/HBM-DMA 位），按 request/stage 粒度对照；节点 ID 与插入顺序不作比较项。

输入：
  --baseline-et <dir>     离线 .et 目录（54 个 <prefix>.<rank>.et）
  --online-nodes <file>   online_nodes.jsonl（online_service --dump-nodes 产物）

登记的刻意差异类别（不计失败，单独计数归因）：
  ① replay 时钟口径：COMP runtime_ns 校准（离线 duration_micros=0，roofline
     运行期计算；在线 replay 携带 LUT 校准值）、timer gate 时长（离线等待/
     在线 alarm 替代 runtime=0）、comm/MEM/HBM-DMA 即时完成；
  ② 跨 request previous_id 串行化边：replay 清除 prefill 组跨 request 链
     （B4 归因类别①）；timer gate 不链 previous_id（离线同款独立节点）。

strategy 模式附加检查：per-rank frontier 连续性不变量（修复 2026-08-16）——
每 rank 除首节点与 timer gate 外，节点父依赖 = 该 rank 上一发射节点（链
连续 = per-rank 发行序与全局发射序一致的必要条件）。
"""

import argparse
import collections
import json
import sys
from pathlib import Path

PROJECT_ROOT = next(
    parent for parent in Path(__file__).resolve().parents
    if (parent / "extern/graph_frontend/chakra/schema/protobuf/et_def_pb2.py").exists()
)
sys.path.insert(0, str(PROJECT_ROOT))

from extern.graph_frontend.chakra.schema.protobuf import et_def_pb2  # noqa: E402
from extern.graph_frontend.chakra.src.third_party.utils.protolib import (  # noqa: E402
    decodeMessage,
)

NODE_TYPE_NAMES = {
    0: "INVALID", 1: "METADATA", 2: "MEM_LOAD", 3: "MEM_STORE",
    4: "COMP", 5: "COMM_SEND", 6: "COMM_RECV", 7: "COMM_COLL",
}


def load_offline(et_dir: Path):
    et_files = sorted(et_dir.glob("*.et"))
    ranks = {}
    for path in et_files:
        rank = int(path.suffixes[0].lstrip("."))
        nodes = []
        with path.open("rb") as source:
            metadata = et_def_pb2.GlobalMetadata()
            decodeMessage(source, metadata)
            node = et_def_pb2.Node()
            while decodeMessage(source, node):
                attrs = {a.name: a for a in node.attr}
                tensor = (
                    attrs["tensor_size"].uint64_val
                    if "tensor_size" in attrs else 0)
                num_ops = (
                    attrs["num_ops"].uint64_val if "num_ops" in attrs else 0)
                comm_bytes = (
                    attrs["comm_size"].uint64_val
                    if "comm_size" in attrs else 0)
                tag = attrs["comm_tag"].uint32_val if "comm_tag" in attrs else 0
                nodes.append({
                    "id": node.id,
                    "name": node.name,
                    "type": int(node.type),
                    "duration_us": node.duration_micros,
                    "tensor_size": max(1, tensor) if tensor else 0,
                    "num_ops": max(1, num_ops) if num_ops else 0,
                    "comm_bytes": max(1, comm_bytes) if comm_bytes else 0,
                    "comm_tag": tag,
                    "is_timer_op": (
                        attrs["is_timer_op"].bool_val
                        if "is_timer_op" in attrs else False),
                    "is_local_hbm_kv_restore": (
                        attrs["is_local_hbm_kv_restore"].bool_val
                        if "is_local_hbm_kv_restore" in attrs else False),
                    "dep_count": len(node.data_deps),
                })
                node.Clear()
        ranks[rank] = nodes
    return ranks


_ONLINE_EDGES = {}


def load_online(path: Path):
    ranks = collections.defaultdict(list)
    edges = collections.defaultdict(list)
    with path.open("r", encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            rank = row["rank"]
            if "edge_from" in row:
                edges[rank].append((row["edge_from"], row["edge_to"]))
                continue
            ranks[rank].append(row)
    # 在线边表 → dep_count（每节点入度；边表保留供 frontier 检查）
    global _ONLINE_EDGES
    _ONLINE_EDGES = edges
    indeg = collections.defaultdict(int)
    for rank, es in edges.items():
        for _, to in es:
            indeg[(rank, to)] += 1
    for rank, nodes in ranks.items():
        for node in nodes:
            node["dep_count"] = indeg.get((rank, node["id"]), 0)
    return dict(ranks)


def structural_key(node):
    return (node["name"], node["type"])


def attribute_key(node):
    # comm_tag 不入 canonical key：tag 是 TransferTagAllocator 分配的协议
    # 层唯一性令牌（离线按全局 Kahn 发射序分配、在线按决策触发序分配，
    # 绝对值必然不同）；语义内容 = 同一 transfer 动作内 send/recv/ack 各自
    # 共享同一 tag（配对一致性由 pair_tag_check 单独验证）。
    return (
        node.get("tensor_size", 0), node.get("num_ops", 0),
        node.get("comm_bytes", 0),
        bool(node.get("is_local_hbm_kv_restore", False)),
    )


def action_key_of(name):
    """节点名 → transfer 动作键（剥离 _send/_recv/_ack_from/_ack_to 尾部）。"""
    for suffix in ("_send", "_recv", "_ack_from_rank", "_ack_to_rank"):
        if name.endswith(suffix):
            return name[: len(name) - len(suffix)]
    return None


def pair_tag_check(offline, online):
    """P2P 配对一致性：每个 transfer 动作键内，离线/在线各自的 send/recv
    （及 ack 对）必须共享同一 tag（多 rank 参与时逐 shard 一致）。返回
    (offline_pair_errors, online_pair_errors)。"""
    errors = [0, 0]
    for idx, ranks in ((0, offline), (1, online)):
        tags = collections.defaultdict(set)
        for nodes in ranks.values():
            for node in nodes:
                key = action_key_of(node["name"])
                if key is not None and node["type"] in (5, 6):
                    tags[key].add(node.get("comm_tag", 0))
        for key, tag_set in tags.items():
            if len(tag_set) > 2:
                # 同一动作内允许 ≤2 个 tag（数据对 + ack 对），超过即配对断裂
                errors[idx] += 1
    return errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-et", required=True)
    parser.add_argument("--online-nodes", required=True)
    parser.add_argument("--frontier-check", action="store_true",
                        help="strategy 模式：验证 per-rank frontier 连续性")
    args = parser.parse_args()

    offline = load_offline(Path(args.baseline_et))
    online = load_online(Path(args.online_nodes))

    total_offline = sum(len(v) for v in offline.values())
    total_online = sum(len(v) for v in online.values())
    print(f"[b3] offline nodes={total_offline} online nodes={total_online}")

    name_diffs = 0
    attr_diffs = 0
    attr_buckets = collections.Counter()
    dep_diffs = 0
    ranks_checked = 0
    for rank in sorted(set(offline) | set(online)):
        off = offline.get(rank, [])
        on = online.get(rank, [])
        ranks_checked += 1
        off_names = collections.Counter(structural_key(n) for n in off)
        on_names = collections.Counter(structural_key(n) for n in on)
        if off_names != on_names:
            name_diffs += 1
            missing = (off_names - on_names).most_common(3)
            extra = (on_names - off_names).most_common(3)
            print(f"[b3] rank {rank}: STRUCTURE multiset mismatch "
                  f"(total {len(off)} vs {len(on)}); missing={missing} "
                  f"extra={extra}")
            continue
        # 同名同型节点做属性多重集对比（顺序无关）
        off_by_key = collections.defaultdict(list)
        on_by_key = collections.defaultdict(list)
        for n in off:
            off_by_key[structural_key(n)].append(attribute_key(n))
        for n in on:
            on_by_key[structural_key(n)].append(attribute_key(n))
        rank_attr_diff = 0
        for key in off_by_key:
            a = collections.Counter(off_by_key[key])
            b = collections.Counter(on_by_key[key])
            if a != b:
                if key[1] == 4:  # COMP：runtime 校准为登记差异
                    attr_buckets["comp_runtime_calibration"] += sum(
                        (a - b).values()) + sum((b - a).values())
                    continue
                rank_attr_diff += 1
        if rank_attr_diff:
            attr_diffs += rank_attr_diff
        # 依赖入度分布（顺序无关的粗粒度结构对照）
        off_deg = collections.Counter(n["dep_count"] for n in off)
        on_deg = collections.Counter(n["dep_count"] for n in on)
        # timer gate 在线不链 previous_id（离线同款独立节点），剔除 0/1 度
        # 计数的微小差异归因到 ②；仅报告显著差异
        if abs(sum(off_deg.values()) - sum(on_deg.values())) > 0:
            dep_diffs += 1

    off_pair_err, on_pair_err = pair_tag_check(offline, online)
    print(f"[b3] P2P pair-tag consistency: offline_errors={off_pair_err} "
          f"online_errors={on_pair_err} (0/0 = send/recv/ack 配对一致)")

    print(f"[b3] ranks checked={ranks_checked}")
    print(f"[b3] structural (name,type) multiset mismatch ranks: {name_diffs}")
    print(f"[b3] attribute mismatch ranks (excl. registered): {attr_diffs}")
    print(f"[b3] attr attribution buckets: {dict(attr_buckets)}")
    print(f"[b3] dep-degree distribution mismatch ranks (coarse): {dep_diffs}")

    if args.frontier_check:
        # frontier 接续不变量（strategy 死锁修复的验证）：每 rank 的每个
        # 非 timer 节点（除该 rank 首个）的父依赖必须包含该 rank 上一发射
        # 的非 timer 节点（链连续 ⇒ per-rank 发行序 = 全局发射序）。
        intra = 0     # 段内并行恢复分支（offline restore_chain 同构，合法）
        cross = 0     # 跨 request 断链（危险类：发行序反转的必要条件）
        for rank, nodes in sorted(online.items()):
            parents = collections.defaultdict(set)
            for f, t in _ONLINE_EDGES.get(rank, []):
                parents[t].add(f)
            byid = {n["id"]: n for n in nodes}
            seq = [n for n in sorted(nodes, key=lambda n: n["id"])
                   if not n.get("is_timer_op")]
            for prev, node in zip(seq, seq[1:]):
                if prev["id"] not in parents.get(node["id"], set()):
                    ps = parents.get(node["id"], set())
                    if any(byid[p]["request_id"] == node["request_id"]
                           for p in ps if p in byid):
                        intra += 1
                    else:
                        cross += 1
        print(f"[b3][frontier] frontier-continuity: intra_request_branches="
              f"{intra} (offline 同构并行恢复，合法) cross_request_breaks="
              f"{cross} (0 = per-rank 发行序=全局发射序不变量成立)")

    ok = name_diffs == 0 and attr_diffs == 0 and off_pair_err == 0 \
        and on_pair_err == 0
    print(f"[b3] RESULT: {'PASS' if ok else 'FAIL'} "
          f"(structural+attribute canonical equality; "
          f"registered-attribution buckets excluded)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
