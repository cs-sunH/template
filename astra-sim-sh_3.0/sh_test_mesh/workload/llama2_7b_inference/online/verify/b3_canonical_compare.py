#!/usr/bin/env python3
"""b3_canonical_compare.py -- sh_3.0 B3 图结构等价对照器（合同⑦ B3）。

比较基准 = 与插入顺序无关的 canonical logical node key：
  (rank, name, type, is_cpu_op, is_timer_op, 规范化属性四元组)
节点 name 内嵌 q{queue_index}_{session}_turn{turn}_{request_id} 前缀与
阶段段（离线 .et 与在线 GraphBatch 同一命名函数产出），故 name+rank+type+
attrs 即语义键；节点 ID 与插入顺序天然不同（离线全局 KV 因果预排序一次
预写 vs 在线按真实阶段完成交错提交），不作比较项（合同⑦）。

边比较：per-rank (from_name -> to_name) 规范化多重集；差异按登记类别
归类（① replay LUT 时钟清除跨 request previous_id 边；② 两段式发射的
emitted-ranks-only 恢复；③ completion 批边界重排）。

用法：
  python3 b3_canonical_compare.py --offline-et <et_prefix_dir> \
      --online <run_dir>/bridge/canonical_nodes.jsonl

离线侧逐 rank 解析 <et_prefix>.{rank}.et（protobuf 直读，属性镜像
TraceBuilder 编码）；在线侧读 SH30_B3_DUMP=1 运行产出的 canonical dump。
"""

import argparse
import collections
import json
import pathlib
import sys

_PROJECT_ROOT = next(
    parent
    for parent in (
        pathlib.Path(__file__).resolve().parent,
        *pathlib.Path(__file__).resolve().parents,
    )
    if (parent / "extern/graph_frontend/chakra/schema/protobuf/et_def_pb2.py").exists()
)
sys.path.insert(0, str(_PROJECT_ROOT))

from extern.graph_frontend.chakra.schema.protobuf import et_def_pb2  # noqa: E402

_TIMER_ATTRS = {"is_timer_op"}


def _proto_node_key(node, rank: int):
    attrs = {}
    for attr in node.attr:
        name = attr.name
        field = attr.WhichOneof("value")
        if field is None:
            value = None
        elif field in ("uint64_val", "int64_val", "uint32_val", "int32_val"):
            value = int(getattr(attr, field))
        elif field == "bool_val":
            value = bool(attr.bool_val)
        elif field == "string_val":
            value = attr.string_val
        elif field == "bool_list":
            value = tuple(attr.bool_list.values)
        else:
            value = None
        attrs[name] = value
    is_timer = bool(attrs.get("is_timer_op", False))
    is_cpu = bool(attrs.get("is_cpu_op", False)) if "is_cpu_op" in attrs else is_timer
    # 类型感知属性键：仅编码该节点类型实际消费的属性（collective 不带
    # comm_src/dst——离线 all_reduce 不写它们；send/recv 不带 collective
    # 字段；comp/mem 各取其消费面），避免缺席-默认值双边不一致。
    ntype = int(node.type)
    if ntype in (5, 6):  # COMM_SEND / COMM_RECV
        canon_attrs = ("comm", (
            int(attrs.get("comm_size", 0) or 0),
            int(attrs.get("comm_src", rank) or 0),
            int(attrs.get("comm_dst", rank) or 0),
            int(attrs.get("comm_tag", 0) or 0),
        ))
    elif ntype == 7:  # COMM_COLL
        canon_attrs = ("coll", (
            int(attrs.get("comm_type", 0) or 0),
            int(attrs.get("comm_size", 0) or 0),
            int(attrs.get("comm_priority", 0) or 0),
            str(attrs.get("pg_name", "") or ""),
            tuple(attrs["involved_dim"]) if "involved_dim" in attrs else (),
        ))
    elif ntype in (2, 3):  # MEM_LOAD / MEM_STORE
        canon_attrs = ("mem", (
            int(attrs.get("tensor_size", 0) or 0),
            bool(attrs.get("is_local_hbm_kv_restore", False)),
        ))
    else:  # COMP / others
        canon_attrs = ("comp", (
            int(attrs.get("num_ops", 0) or 0),
            int(attrs.get("tensor_size", 0) or 0),
            int(attrs.get("remote_weight_bytes", 0) or 0),
        ))
    return (rank, node.name, ntype, is_cpu, is_timer, canon_attrs)


def _online_node_key(row):
    # 与 _proto_node_key 相同的类型感知属性面。
    ntype = int(row["type"])
    compute = row.get("compute", {})
    mem = row.get("mem", {})
    comm = row.get("comm", {})
    coll = row.get("coll", {})
    if ntype in (5, 6):
        canon_attrs = ("comm", (
            int(comm.get("bytes", 0) or 0),
            int(comm.get("src", row["rank"]) or 0),
            int(comm.get("dst", row["rank"]) or 0),
            int(comm.get("tag", 0) or 0),
        ))
    elif ntype == 7:
        canon_attrs = ("coll", (
            int(coll.get("comm_type", 0) or 0),
            int(coll.get("bytes", 0) or 0),
            int(coll.get("priority", 0) or 0),
            str(coll.get("pg_name", "") or ""),
            tuple(coll.get("involved_dim", ())),
        ))
    elif ntype in (2, 3):
        canon_attrs = ("mem", (
            int(mem.get("tensor_size", 0) or 0)
            or int(compute.get("tensor_size", 0) or 0),
            bool(mem.get("is_local_hbm_kv_restore", False)),
        ))
    else:
        canon_attrs = ("comp", (
            int(compute.get("num_ops", 0) or 0),
            int(compute.get("tensor_size", 0) or 0),
            int(compute.get("remote_weight_bytes", 0) or 0),
        ))
    return (row["rank"], row["name"], ntype,
            bool(row.get("is_cpu_op", False)),
            bool(row.get("is_timer_op", False)), canon_attrs)


def _iter_et_messages(buffer: bytes):
    """Chakra ET v3 length-delimited decode (protolib encodeMessage =
    varint length + serialized message, repeated: GlobalMetadata first, then
    Nodes)."""
    import google.protobuf.internal.decoder as _dec
    offset = 0
    while offset < len(buffer):
        msg_len, new_offset = _dec._DecodeVarint32(buffer, offset)
        yield buffer[new_offset:new_offset + msg_len]
        offset = new_offset + msg_len


def _iter_et_nodes(buffer: bytes):
    messages = list(_iter_et_messages(buffer))
    for payload in messages[1:]:  # [0] = GlobalMetadata
        node = et_def_pb2.Node()
        node.ParseFromString(payload)
        yield node


def load_offline(et_dir: pathlib.Path, output_prefix: str, npus: int):
    nodes = collections.Counter()
    names_by_rank = collections.defaultdict(dict)
    edges = collections.Counter()
    for rank in range(npus):
        path = et_dir / f"{output_prefix}.{rank}.et"
        with open(path, "rb") as source:
            buffer = source.read()
        for node in _iter_et_nodes(buffer):
            nodes[_proto_node_key(node, rank)] += 1
            names_by_rank[rank][node.id] = node.name
        for node in _iter_et_nodes(buffer):
            for parent in list(node.data_deps) + list(node.ctrl_deps):
                edges[(rank,
                       names_by_rank[rank].get(parent, f"id{parent}"),
                       node.name)] += 1
    return nodes, edges


def load_online(dump_path: pathlib.Path):
    nodes = collections.Counter()
    edges = collections.Counter()
    with open(dump_path, "r", encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            if row["kind"] == "node":
                nodes[_online_node_key(row)] += 1
            else:
                edges[(row["rank"], row["from_name"], row["to_name"])] += 1
    return nodes, edges


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline-et", required=True,
                        help="离线 .et 目录（含 <prefix>.{rank}.et）")
    parser.add_argument("--prefix", default="llama2_7b_inference")
    parser.add_argument("--npus", type=int, default=54)
    parser.add_argument("--online", required=True,
                        help="SH30_B3_DUMP=1 运行产出的 canonical_nodes.jsonl")
    args = parser.parse_args()

    off_nodes, off_edges = load_offline(
        pathlib.Path(args.offline_et), args.prefix, args.npus)
    on_nodes, on_edges = load_online(pathlib.Path(args.online))

    node_missing = off_nodes - on_nodes
    node_extra = on_nodes - off_nodes
    edge_missing = off_edges - on_edges
    edge_extra = on_edges - off_edges
    print(f"[B3] offline nodes={sum(off_nodes.values())} "
          f"online nodes={sum(on_nodes.values())}")
    print(f"[B3] node multiset: missing={sum(node_missing.values())} "
          f"extra={sum(node_extra.values())}")
    for key, count in list(node_missing.items())[:10]:
        print("  MISSING", count, key[0], key[1], "type", key[2])
    for key, count in list(node_extra.items())[:10]:
        print("  EXTRA  ", count, key[0], key[1], "type", key[2])
    print(f"[B3] offline edges={sum(off_edges.values())} "
          f"online edges={sum(on_edges.values())} "
          f"missing={sum(edge_missing.values())} "
          f"extra={sum(edge_extra.values())}")
    # 归因：跨 request previous_id 边（name 前缀不同的 from/to）
    def cross(edge):
        rank, frm, to = edge
        def pfx(n):
            return n.split("_turn")[0] if n and "_turn" in n else n
        return frm and to and pfx(frm) != pfx(to)
    cat_cross_missing = sum(c for e, c in edge_missing.items() if cross(e))
    cat_cross_extra = sum(c for e, c in edge_extra.items() if cross(e))
    print(f"[B3] edge diff categories: cross-request missing="
          f"{cat_cross_missing} extra={cat_cross_extra}; "
          f"within-request missing="
          f"{sum(edge_missing.values()) - cat_cross_missing} extra="
          f"{sum(edge_extra.values()) - cat_cross_extra}")
    ok_nodes = not node_missing and not node_extra
    print(f"[B3] node canonical multiset equal: {ok_nodes}")
    return 0 if ok_nodes else 1


if __name__ == "__main__":
    sys.exit(main())
