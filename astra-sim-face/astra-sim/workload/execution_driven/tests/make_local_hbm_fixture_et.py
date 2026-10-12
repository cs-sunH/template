#!/usr/bin/env python3
"""make_local_hbm_fixture_et.py -- multi-user local-HBM contention fixture.

Writes a self-contained 2-rank static-simulation input set into
  sh_test_mesh/generated/local_hbm_fixture/
consumed by AstraSim_Analytical_Congestion_Aware_LocalHbmTest
(local_hbm_bandwidth_model_test.cc):

  fixture.0.et / fixture.1.et   dependency-free p2p + comp graph (below)
  system.json                   roofline on, local-mem-bw 6 GB/s (=6 B/ns),
                                local-mem-latency 100 ns, peak-perf 1 TFLOPS,
                                hbm-bandwidth-contention on
  fast/network.yml              link 100 GB/s, latency 20 ns
  slow/network.yml              link 0.5 GB/s, latency 20 ns
  legacy/system.json            same as system.json with
                                hbm-bandwidth-contention = 0 (uses the fast
                                network; verifies the full legacy fallback)

Graph (all nodes dependency-free; issue order = ascending id; the static
HardwareResource single comm slot delays rank0 node 4 until node 2 frees it):

  rank 0: id 1 COMP(num_ops=1, tensor_size=30)
          id 2 COMM_SEND -> rank1, 60 B,  tag 1
          id 3 COMM_RECV <- rank1, 120 B, tag 2
          id 4 COMM_SEND -> rank1, 40 B,  tag 3, hbm-charge=false
  rank 1: id 1 COMM_SEND -> rank0, 120 B, tag 2
          id 2 COMM_RECV <- rank0, 60 B,  tag 1
          id 3 COMM_RECV <- rank0, 40 B,  tag 3, hbm-charge=false

Expected timelines (full numeric derivations live in
local_hbm_bandwidth_model_test.cc): rank0 runs the 3-user case
(COMP 30 B + COMM_READ 60 B + COMM_WRITE 120 B), rank1 the 2-user case;
the fast network completes the network side long before the HBM jobs
(network-first join order) and the slow network after them (HBM-first
order), so both join orders are exercised.

The generated/ directory is gitignored by design; rerun this script to
regenerate the fixture.
"""

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
    COMP_NODE,
    AttributeProto as ChakraAttr,
    GlobalMetadata,
    Node as ChakraNode,
)
from extern.graph_frontend.chakra.src.third_party.utils.protolib import (  # noqa: E402
    encodeMessage as encode_message,
)

OUTPUT_DIR = PROJECT_ROOT / "sh_test_mesh" / "generated" / "local_hbm_fixture"
TRACE_PREFIX = "fixture"

SYSTEM_JSON = """{
    "scheduling-policy": "FIFO",
    "peak-perf": 1,
    "roofline-enabled": 1,
    "local-mem-bw": 6,
    "local-mem-latency": 100,
    "hbm-bandwidth-contention": 1
}
"""

# legacy scenario (hbm-bandwidth-contention off): full fallback to the
# closed-form roofline COMP duration and network-only comm completion.
SYSTEM_JSON_LEGACY = """{
    "scheduling-policy": "FIFO",
    "peak-perf": 1,
    "roofline-enabled": 1,
    "local-mem-bw": 6,
    "local-mem-latency": 100,
    "hbm-bandwidth-contention": 0
}
"""


FAST_NETWORK_YML = (
    "topology: [ FullyConnected ]\n"
    "npus_count: [ 2 ]\n"
    "bandwidth: [ 100 ]\n"
    "latency: [ 20 ]\n"
)

SLOW_NETWORK_YML = (
    "topology: [ FullyConnected ]\n"
    "npus_count: [ 2 ]\n"
    "bandwidth: [ 0.5 ]\n"
    "latency: [ 20 ]\n"
)


def uint64_attr(name: str, value: int) -> ChakraAttr:
    return ChakraAttr(name=name, uint64_val=max(1, int(value)))


def comm_node(node_id: int, name: str, node_type: int, src: int, dst: int,
              comm_size: int, tag: int, hbm_charge: bool) -> ChakraNode:
    node = ChakraNode()
    node.id = node_id
    node.name = name
    node.type = node_type
    node.attr.extend([
        ChakraAttr(name="comm_src", uint32_val=src),
        ChakraAttr(name="comm_dst", uint32_val=dst),
        uint64_attr("comm_size", comm_size),
        ChakraAttr(name="comm_tag", uint32_val=tag),
        ChakraAttr(name="hbm-charge", bool_val=hbm_charge),
    ])
    return node


def build_rank0() -> list[ChakraNode]:
    comp = ChakraNode()
    comp.id = 1
    comp.name = "hbm_comp"
    comp.type = COMP_NODE
    comp.attr.append(uint64_attr("num_ops", 1))
    comp.attr.append(uint64_attr("tensor_size", 30))

    send_60 = comm_node(2, "hbm_send_60", COMM_SEND_NODE, 0, 1, 60, 1, True)
    recv_120 = comm_node(3, "hbm_recv_120", COMM_RECV_NODE, 1, 0, 120, 2, True)
    send_40_uncharged = comm_node(
        4, "hbm_send_40_uncharged", COMM_SEND_NODE, 0, 1, 40, 3, False)
    return [comp, send_60, recv_120, send_40_uncharged]


def build_rank1() -> list[ChakraNode]:
    send_120 = comm_node(1, "hbm_send_120", COMM_SEND_NODE, 1, 0, 120, 2, True)
    recv_60 = comm_node(2, "hbm_recv_60", COMM_RECV_NODE, 0, 1, 60, 1, True)
    recv_40_uncharged = comm_node(
        3, "hbm_recv_40_uncharged", COMM_RECV_NODE, 0, 1, 40, 3, False)
    return [send_120, recv_60, recv_40_uncharged]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "fast").mkdir(exist_ok=True)
    (OUTPUT_DIR / "slow").mkdir(exist_ok=True)
    (OUTPUT_DIR / "legacy").mkdir(exist_ok=True)

    (OUTPUT_DIR / "system.json").write_text(SYSTEM_JSON, encoding="utf-8")
    (OUTPUT_DIR / "legacy" / "system.json").write_text(
        SYSTEM_JSON_LEGACY, encoding="utf-8")
    (OUTPUT_DIR / "fast" / "network.yml").write_text(
        FAST_NETWORK_YML, encoding="utf-8")
    (OUTPUT_DIR / "slow" / "network.yml").write_text(
        SLOW_NETWORK_YML, encoding="utf-8")

    for rank, nodes in ((0, build_rank0()), (1, build_rank1())):
        path = OUTPUT_DIR / f"{TRACE_PREFIX}.{rank}.et"
        with path.open("wb") as output:
            metadata = GlobalMetadata(version="0.0.4")
            metadata.attr.extend([
                ChakraAttr(name="schema", string_val="1.0.2-chakra.0.0.4"),
                ChakraAttr(name="execution_mode",
                           string_val="local_hbm_fixture"),
                ChakraAttr(name="trace_granularity", string_val="synthetic"),
            ])
            encode_message(output, metadata)
            for node in nodes:
                encode_message(output, node)
        print(f"[fixture] wrote {path}")

    print(f"[fixture] config set complete in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
