#!/usr/bin/env python3
"""make_remote_port_static_fixture_et.py -- static/ETFeeder remote-port
issue-gating fixture generator (SerDes 片外链路并发化改造执行方案 V5.3,
stage 5.4).

Contract (fixed; the workflow runs exactly this):
  python3 make_remote_port_static_fixture_et.py --out-dir <temporary dir>
and points the test binary at the SAME directory:
  AstraSim_Analytical_Congestion_Aware_RemotePortStaticGateTest \
      --fixture-dir <the same temporary dir>

Products written into --out-dir:
  remote_port_static_gate.0.et   rank0: two dependency-free remote MEM loads
                                 plus one COMM_SEND (below)
  remote_port_static_gate.1.et   rank1: the matching COMM_RECV
  system.json                    official template shape (scheduling-policy /
                                 preferred-dataset-splits /
                                 collective-optimization present, all four
                                 *-implementation keys ["ring"] -- one per
                                 physical dimension of the single-dimension
                                 2-NPU Mesh network; no
                                 *-implementation-custom, no
                                 doubleBinaryTree, no local-mem keys so the
                                 N-way local-HBM model auto-disables)
  remote_memory.json             PER_NPU ports on ranks 0 and 1, bw 6 B/ns,
                                 latency 100 ns
  network.yml                    2-NPU Mesh, 4050 GB/s links (SI identity
                                 1 GB/s = 1 B/ns), 25 ns

Graph (all nodes dependency-free; the first issue_dep_free_nodes scan issues
everything in ascending id order):

  rank 0: id 1 MEM_LOAD tensor_size=600  ("static_mem_a")
          id 2 MEM_LOAD tensor_size=600  ("static_mem_b")
          id 3 COMM_SEND -> rank1, 1000000 B, tag 7, hbm-charge=false
  rank 1: id 1 COMM_RECV <- rank0, 1000000 B, tag 7, hbm-charge=false

Gate semantics under test (HardwareResource.cc classify(), plan sec.4):
both MEM loads occupy the independent count-based remote-MEM slot and MUST
both issue from the same scan even though id 1 already holds it; the
COMM_SEND takes the legacy comm single slot without blocking either MEM.
Expected timing (analytical remote-port model, derived in
remote_port_static_gate_test.cc): the two 600 B MEMs share the port at
3 B/ns over [100,300) -> both callbacks at Tick 300; the p2p completion
follows the network (~247 ns transfer + 25 ns latency), asserted only as
"completed exactly once".
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
    COMM_RECV_NODE,
    COMM_SEND_NODE,
    MEM_LOAD_NODE,
    AttributeProto as ChakraAttr,
    GlobalMetadata,
    Node as ChakraNode,
)
from extern.graph_frontend.chakra.src.third_party.utils.protolib import (  # noqa: E402
    encodeMessage as encode_message,
)

TRACE_PREFIX = "remote_port_static_gate"

SYSTEM_JSON = """{
    "scheduling-policy": "LIFO",
    "endpoint-delay": 1,
    "active-chunks-per-dimension": 1,
    "preferred-dataset-splits": 1,
    "all-reduce-implementation": ["ring"],
    "all-gather-implementation": ["ring"],
    "reduce-scatter-implementation": ["ring"],
    "all-to-all-implementation": ["ring"],
    "collective-optimization": "localBWAware"
}
"""

REMOTE_MEMORY_JSON = """{
    "memory-type": "PER_NPU_MEMORY_EXPANSION",
    "npu-ids": [0, 1],
    "remote-mem-bw": 6,
    "remote-mem-latency": 100
}
"""

NETWORK_YML = (
    "topology: [ Mesh ]\n"
    "npus_count: [ 2 ]\n"
    "bandwidth: [ 4050 ]\n"
    "latency: [ 25 ]\n"
)

P2P_BYTES = 1000000
P2P_TAG = 7


def uint64_attr(name: str, value: int) -> ChakraAttr:
    return ChakraAttr(name=name, uint64_val=max(1, int(value)))


def mem_load(node_id: int, name: str, tensor_size: int) -> ChakraNode:
    node = ChakraNode()
    node.id = node_id
    node.name = name
    node.type = MEM_LOAD_NODE
    node.attr.append(uint64_attr("tensor_size", tensor_size))
    return node


def comm_node(node_id: int, name: str, node_type: int, src: int, dst: int,
              comm_size: int, tag: int) -> ChakraNode:
    node = ChakraNode()
    node.id = node_id
    node.name = name
    node.type = node_type
    node.attr.extend([
        ChakraAttr(name="comm_src", uint32_val=src),
        ChakraAttr(name="comm_dst", uint32_val=dst),
        uint64_attr("comm_size", comm_size),
        ChakraAttr(name="comm_tag", uint32_val=tag),
        ChakraAttr(name="hbm-charge", bool_val=False),
    ])
    return node


def build_rank0() -> list:
    # Two dependency-free remote MEM loads: under the reworked gate both
    # issue from one scan (the remote-MEM slot never blocks); under the old
    # comm-single-slot behavior id 2 would have been stuck behind id 1.
    mem_a = mem_load(1, "static_mem_a", 600)
    mem_b = mem_load(2, "static_mem_b", 600)
    send = comm_node(3, "static_send", COMM_SEND_NODE, 0, 1, P2P_BYTES,
                     P2P_TAG)
    return [mem_a, mem_b, send]


def build_rank1() -> list:
    recv = comm_node(1, "static_recv", COMM_RECV_NODE, 0, 1, P2P_BYTES,
                     P2P_TAG)
    return [recv]


def build_metadata() -> GlobalMetadata:
    metadata = GlobalMetadata(version="0.0.4")
    metadata.attr.extend([
        ChakraAttr(name="schema", string_val="1.0.2-chakra.0.0.4"),
        ChakraAttr(name="execution_mode", string_val="remote_port_static"),
        ChakraAttr(name="trace_granularity", string_val="synthetic"),
    ])
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the static/ETFeeder remote-port gate fixture.")
    parser.add_argument("--out-dir", required=True,
                        help="directory receiving the .et files and configs")
    args = parser.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "system.json").write_text(SYSTEM_JSON, encoding="utf-8")
    (out_dir / "remote_memory.json").write_text(
        REMOTE_MEMORY_JSON, encoding="utf-8")
    (out_dir / "network.yml").write_text(NETWORK_YML, encoding="utf-8")

    for rank, nodes in ((0, build_rank0()), (1, build_rank1())):
        path = out_dir / f"{TRACE_PREFIX}.{rank}.et"
        with path.open("wb") as output:
            encode_message(output, build_metadata())
            for node in nodes:
                encode_message(output, node)
        print(f"[remote_port_static_fixture] wrote {path}")

    print(f"[remote_port_static_fixture] fixture complete in {out_dir}; "
          f"run the test binary with --fixture-dir {out_dir}")


if __name__ == "__main__":
    main()
