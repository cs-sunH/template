#!/usr/bin/env python3
"""make_remote_port_static_fixture_et.py -- static/ETFeeder remote-port gate
fixture generator (SerDes片外链路并发化改造执行方案 V5.3 阶段 5.4; test-only).

Fixed contract (plan 阶段 5.7):
    python3 make_remote_port_static_fixture_et.py --out-dir <临时目录>

Writes into <out-dir>:
    fixture.0.et          rank0: TWO dependency-free plain MEM_LOAD nodes
                          (600 B each, ids 0/1) -- the double-MEM gate
                          subject.  No "is_local_hbm_kv_restore", no
                          "hbm-access-mode": both fall through to the
                          remote-memory port transaction path.
    fixture.1.et          rank1: ONE plain MEM_LOAD 600 B (id 0) -- the
                          cross-port control.
    system.json           official template shape (preferred-dataset-splits /
                          scheduling-policy / collective-optimization
                          present; the four *-implementation keys all
                          ["ring","ring"]; roofline/replay/contention off).
                          The joint deep-dive H1/H2/H3 landmines (custom
                          impls, doubleBinaryTree, dimensionless comm
                          groups) are never armed.
    comm_group.json       "{}" -- the fixture has no collectives, and an
                          empty group set cannot carry a dimensions-less
                          group (same precedent as make_hbm_nway_fixture.py).
    network.yml           synthetic 2-rank line topology.
    remote_memory.json    PER_NPU ports on ranks {0,1}: remote-mem-bw 6 B/ns,
                          remote-mem-latency 100 ns -- every runtime instant
                          hand-computable: shared 600 B flows stream at
                          3 B/ns over [100,300] -> both rank0 terminals at
                          t=300 (the retired comm-single-slot gate would
                          serialize them to 200/400); the control finishes
                          solo at t=200.

The companion binary remote_port_static_gate_test.cc consumes this directory
via --fixture-dir <同一临时目录>.
"""

import argparse
import json
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
    MEM_LOAD_NODE,
    AttributeProto as ChakraAttr,
    GlobalMetadata,
    Node as ChakraNode,
)
from extern.graph_frontend.chakra.src.third_party.utils.protolib import (  # noqa: E402
    encodeMessage as encode_message,
)

TRACE_PREFIX = "fixture"

# Fixture physics (a GB/s value is consumed as B/ns by the engine;
# 6 B/ns per PER_NPU port, 100 ns fixed latency).
REMOTE_MEM_BW = 6
REMOTE_MEM_LATENCY_NS = 100


def uint64_attr(name: str, value: int) -> ChakraAttr:
    return ChakraAttr(name=name, uint64_val=value)


def build_metadata() -> GlobalMetadata:
    metadata = GlobalMetadata(version="0.0.4")
    metadata.attr.extend(
        [
            ChakraAttr(name="schema", string_val="1.0.2-chakra.0.0.4"),
            ChakraAttr(name="execution_mode",
                       string_val="remote_port_static_gate_fixture"),
            ChakraAttr(name="trace_granularity", string_val="synthetic"),
        ]
    )
    return metadata


def remote_mem_node(node_id: int, name: str, tensor_size: int) -> ChakraNode:
    node = ChakraNode()
    node.id = node_id
    node.name = name
    node.type = MEM_LOAD_NODE
    node.attr.append(uint64_attr("tensor_size", tensor_size))
    return node


def rank0_nodes() -> list[ChakraNode]:
    # The gate subject: TWO dependency-free remote MEM nodes on ONE rank.
    # Both are issuable at tick 0; the retired comm-single-slot gate would
    # have held the second until the first released the slot.
    return [
        remote_mem_node(0, "r0_remote_mem_a", 600),
        remote_mem_node(1, "r0_remote_mem_b", 600),
    ]


def rank1_nodes() -> list[ChakraNode]:
    return [
        remote_mem_node(0, "r1_remote_mem_control", 600),
    ]


def base_system() -> dict:
    return {
        "scheduling-policy": "LIFO",
        "endpoint-delay": 10,
        "active-chunks-per-dimension": 1,
        "preferred-dataset-splits": 6,
        "all-reduce-implementation": ["ring", "ring"],
        "all-gather-implementation": ["ring", "ring"],
        "reduce-scatter-implementation": ["ring", "ring"],
        "all-to-all-implementation": ["ring", "ring"],
        "collective-optimization": "localBWAware",
        "roofline-enabled": 0,
        "replay-only": 0,
        "track-local-mem": 0,
        "trace-enabled": 0,
        "hbm-bandwidth-contention": 0,
        "peak-perf": 1000,
        "local-mem-bw": 1640.0,
        "local-mem-latency": 100,
        "remote-mem-bw": REMOTE_MEM_BW,
        "remote-mem-latency": REMOTE_MEM_LATENCY_NS,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the static remote-port gate fixture "
        "(same-rank double MEM) into --out-dir.")
    parser.add_argument("--out-dir", required=True,
                        help="target directory (temporary; created if "
                        "missing)")
    args = parser.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for rank, builder in ((0, rank0_nodes), (1, rank1_nodes)):
        path = out_dir / f"{TRACE_PREFIX}.{rank}.et"
        with path.open("wb") as output:
            encode_message(output, build_metadata())
            for node in builder():
                encode_message(output, node)
        print(f"[remote_port_static_gate_fixture] wrote {path}")

    (out_dir / "system.json").write_text(
        json.dumps(base_system(), indent=2) + "\n", encoding="utf-8")
    (out_dir / "comm_group.json").write_text("{}\n", encoding="utf-8")
    (out_dir / "network.yml").write_text(
        "# remote_port_static_gate synthetic 2-rank line topology\n"
        "topology: [ Line, Line ]\n"
        "npus_count: [ 2, 1 ]\n"
        "bandwidth: [ 4000.0, 4000.0 ]\n"
        "latency: [ 5, 5 ]\n",
        encoding="utf-8")
    (out_dir / "remote_memory.json").write_text(
        json.dumps(
            {
                "memory-type": "PER_NPU_MEMORY_EXPANSION",
                "remote-mem-bw": REMOTE_MEM_BW,
                "remote-mem-latency": REMOTE_MEM_LATENCY_NS,
                "npu-ids": [0, 1],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8")

    print(
        f"[remote_port_static_gate_fixture] generated fixture in {out_dir} "
        f"(rank0: 2 dep-free 600 B MEMs; rank1: 1 control 600 B MEM); run "
        f"the static gate test with --fixture-dir {out_dir}")


if __name__ == "__main__":
    main()
