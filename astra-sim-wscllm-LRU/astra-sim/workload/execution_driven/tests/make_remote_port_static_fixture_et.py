#!/usr/bin/env python3
"""make_remote_port_static_fixture_et.py -- 《SerDes片外链路并发化改造执行方案》
阶段 5.4 static 门控夹具生成器（wscllm-LRU，2026-09-24）。

契约（方案 阶段 5.7）：python3 make_remote_port_static_fixture_et.py --out-dir
<临时目录>；静态测试二进制 AstraSim_Analytical_Congestion_Aware_
RemotePortStaticGateTest 接受同一目录 --fixture-dir <临时目录>。

产物（全部写入 --out-dir，幂等——重复运行覆盖同名文件、内容确定）：
  remote_port_static_gate.0.et  rank0：两个无依赖远端 MEM_LOAD
                                （id=1 300B、id=2 600B；不带
                                is_local_hbm_kv_restore / hbm-access-mode
                                属性 => 纯远端端口流量），
                                同 rank 无 Data 依赖（首次扫描同发）
  remote_port_static_gate.1..5.et  rank1-5：GlobalMetadata + 一个
                                INVALID_NODE（skip_invalid -> Skipped 终态；
                                ETFeeder 构造期要求至少一个无依赖节点，
                                不产生远端事务）
  system.json / comm_group.json / network.yml / remote_memory.json
                                官方模板形状配置（阶段 5 避雷：三键齐备、
                                四个 *-implementation 全 ["ring","ring"]、
                                comm_group 组带 ranks+dimensions；禁
                                custom/doubleBinaryTree/无 dimensions 组）。

数学口径（静态门控断言用）：remote-mem-bw 6 B/ns、remote-mem-latency 100ns、
PER_NPU；t0 发射后 [t0+100,t0+200) 双流各 3B/ns（300B 流 t0+200 完成、
shared 窗口 100ns），幸存 600B 流重分 6B/ns 于 t0+250 完成。
"""

import argparse
import pathlib
import sys

PROJECT_ROOT = next(
    parent
    for parent in (
        pathlib.Path(__file__).resolve().parent,
        *pathlib.Path(__file__).resolve().parents,
    )
    if (parent / "extern/graph_frontend/chakra/schema/protobuf/et_def_pb2.py").exists()
)
sys.path.insert(0, str(PROJECT_ROOT))

from extern.graph_frontend.chakra.schema.protobuf.et_def_pb2 import (  # noqa: E402
    AttributeProto as ChakraAttr,
    GlobalMetadata,
    Node as ChakraNode,
)
from extern.graph_frontend.chakra.src.third_party.utils.protolib import (  # noqa: E402
    encodeMessage as encode_message,
)

ET_PREFIX = "remote_port_static_gate"
RANKS = 6


def uint64_attr(name: str, value: int) -> ChakraAttr:
    return ChakraAttr(name=name, uint64_val=max(1, int(value)))


def build_metadata() -> GlobalMetadata:
    metadata = GlobalMetadata(version="0.0.4")
    metadata.attr.extend(
        [
            ChakraAttr(name="schema", string_val="1.0.2-chakra.0.0.4"),
            ChakraAttr(name="execution_mode", string_val="static_gate_fixture"),
            ChakraAttr(name="trace_granularity", string_val="synthetic"),
        ]
    )
    return metadata


def mem_node(node_id: int, tensor_size: int, name: str) -> ChakraNode:
    # 远端 MEM：无 is_local_hbm_kv_restore / hbm-access-mode 属性 =>
    # classify_hw_resource -> RemoteMem（计数制、无上限、不占 comm 槽）。
    node = ChakraNode()
    node.id = node_id
    node.name = name
    node.type = 2  # MEM_LOAD_NODE（et_def.proto:111）
    node.attr.append(ChakraAttr(name="is_cpu_op", bool_val=False))
    node.attr.append(uint64_attr("tensor_size", tensor_size))
    return node


def invalid_node(node_id: int, name: str) -> ChakraNode:
    # 空闲填充节点：INVALID_NODE 走 skip_invalid -> Skipped 终态，不占任何
    # 资源/端口。ETFeeder 构造期 resolve_dependancy_free_nodes 会对零空闲
    # 节点的迹抛 "No dependancy free nodes found"（et_feeder.cpp:71 /
    # dependancy_solver.cpp:104-108），因此每个 rank 的迹至少要有一个
    # 无依赖节点——ranks 1-5 用它占位。
    node = ChakraNode()
    node.id = node_id
    node.name = name
    node.type = 0  # INVALID_NODE（et_def.proto:109）
    node.attr.append(ChakraAttr(name="is_cpu_op", bool_val=False))
    return node


def write_rank_trace(out_dir: pathlib.Path, rank: int) -> str:
    path = out_dir / f"{ET_PREFIX}.{rank}.et"
    with path.open("wb") as output:
        encode_message(output, build_metadata())
        if rank == 0:
            # 同 rank 两个无 Data 依赖的远端 MEM：首扫描即同发。
            for node in (mem_node(1, 300, "gateA_mem_300B"),
                         mem_node(2, 600, "gateA_mem_600B")):
                encode_message(output, node)
        else:
            # 非 0 rank：至少一个无依赖节点，否则 ETFeeder 构造期抛
            # "No dependancy free nodes found"（见 invalid_node 注释）。
            encode_message(output, invalid_node(0, f"filler_invalid_{rank}"))
    return str(path)


def write_configs(out_dir: pathlib.Path) -> list:
    written = []
    system_json = """{
  "scheduling-policy": "LIFO",
  "endpoint-delay": 10,
  "active-chunks-per-dimension": 1,
  "preferred-dataset-splits": 6,
  "all-reduce-implementation": ["ring", "ring"],
  "all-gather-implementation": ["ring", "ring"],
  "reduce-scatter-implementation": ["ring", "ring"],
  "all-to-all-implementation": ["ring", "ring"],
  "collective-optimization": "localBWAware",
  "boost-mode": 0,
  "roofline-enabled": 1,
  "replay-only": 0,
  "track-local-mem": 0,
  "trace-enabled": 0,
  "hbm-bandwidth-contention": 0,
  "hbm-kv-restore-bandwidth-sharing": 0,
  "peak-perf": 261.12,
  "local-mem-bw": 1640.0,
  "local-mem-latency": 100,
  "remote-mem-bw": 6.0,
  "remote-mem-latency": 100
}
"""
    comm_group_json = """{
  "1": {
    "ranks": [0, 1, 2, 3, 4, 5],
    "dimensions": [3, 2]
  }
}
"""
    network_yml = """topology: [ Mesh, Mesh ]
npus_count: [ 3, 2 ]
bandwidth: [ 4050, 4050 ]
latency: [ 25, 25 ]
"""
    remote_memory_json = """{
  "memory-type": "PER_NPU_MEMORY_EXPANSION",
  "npu-ids": [0, 1, 2, 3, 4, 5],
  "remote-mem-bw": 6.0,
  "remote-mem-latency": 100
}
"""
    for name, content in (
        ("system.json", system_json),
        ("comm_group.json", comm_group_json),
        ("network.yml", network_yml),
        ("remote_memory.json", remote_memory_json),
    ):
        path = out_dir / name
        path.write_text(content, encoding="utf-8")
        written.append(str(path))
    return written


def main() -> int:
    parser = argparse.ArgumentParser(
        description="static remote-port gate fixture generator (stage 5.4)"
    )
    parser.add_argument("--out-dir", required=True,
                        help="output directory (typically a temp dir)")
    args = parser.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    written = []
    for rank in range(RANKS):
        written.append(write_rank_trace(out_dir, rank))
    written.extend(write_configs(out_dir))

    for path in written:
        print(f"[remote_port_static_fixture] wrote {path}")
    print(
        f"[remote_port_static_fixture] {RANKS} rank traces + 4 configs in "
        f"{out_dir}; rank0 = 2 dependency-free remote MEMs (300B id=1, "
        "600B id=2); expected concurrent issue = 2 on port0"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
