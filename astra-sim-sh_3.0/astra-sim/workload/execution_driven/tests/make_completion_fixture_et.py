#!/usr/bin/env python3
"""make_completion_fixture_et.py -- step-1-3 synthetic .et generator.

Writes one tiny chakra .et per rank into
  sh_test_mesh/generated/completion_fixture/fixture.{rank}.et
with a 5-node dependency-free graph that exercises all terminal insertion
sites of Workload::call / skip_invalid (static path; the two MEM nodes cover
the sh_3.0-specific remote-FIFO and local-HBM completion chains):

  node 0 INVALID_NODE   -> skip_invalid          (Skipped,   site 1)
  node 1 COMP_NODE      -> generic wlhd branch   (Success,   site 2)
  node 2 COMM_COLL_NODE -> collective branch     (Success,   site 3)
  node 3 MEM_LOAD (remote FIFO) -> generic wlhd branch via
         AnalyticalRemoteMemory completion (Success, sh_3.0-specific chain)
  node 4 MEM_LOAD (is_local_hbm_kv_restore) -> generic wlhd branch via
         LocalHbmBandwidthModel completion (Success, sh_3.0-specific chain)

The nodes are intentionally dependency-free: they all issue in the first
issue_dep_free_nodes scan, and the completion events keep the event queue
alive. (A chained graph would dead-end after skip_invalid -- that path
never re-triggers issue_dep_free_nodes, matching how real traces place
invalid/metadata nodes at chain ends.)

The all_reduce uses the rank's own pg from the fixed comm_group.json
(runtime_config/face_case5_config_c__validation-160gib__edge_remote_memory_pool),
exactly like the real generated traces. The node attrs mirror the encoding
of TraceBuilder in generate_trace.py (comp/all_reduce), so ETFeeder parses
them identically.

The generated/ directory is gitignored by design (byte-equivalence gates);
rerun this script to regenerate the fixture trace. The expectation
(54 ranks x 5 nodes on port ranks / 4 on non-port ranks = 242 completions) is asserted by
completion_observer_fixture_main.cc.
"""

import hashlib
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
    ALL_REDUCE,
    COMM_COLL_NODE,
    MEM_LOAD_NODE,
    COMP_NODE,
    AttributeProto as ChakraAttr,
    BoolList,
    GlobalMetadata,
    Node as ChakraNode,
)
from extern.graph_frontend.chakra.src.third_party.utils.protolib import (  # noqa: E402
    encodeMessage as encode_message,
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "sh_test_mesh"
    / "generated"
    / "completion_fixture"
)
TRACE_PREFIX = "fixture"
COMM_GROUP = (
    PROJECT_ROOT
    / "sh_test_mesh"
    / "generated"
    / "runtime_config"
    / "face_case5_config_c__validation-160gib__edge_remote_memory_pool"
    / "comm_group.json"
)


def uint64_attr(name: str, value: int) -> ChakraAttr:
    return ChakraAttr(name=name, uint64_val=max(1, int(value)))


def build_metadata() -> GlobalMetadata:
    metadata = GlobalMetadata(version="0.0.4")
    metadata.attr.extend(
        [
            ChakraAttr(name="schema", string_val="1.0.2-chakra.0.0.4"),
            ChakraAttr(name="execution_mode", string_val="completion_fixture"),
            ChakraAttr(name="trace_granularity", string_val="synthetic"),
        ]
    )
    return metadata


def main() -> None:
    with open(COMM_GROUP, encoding="utf-8") as source:
        comm_group = json.load(source)
    pg_for_rank = {}
    for pg_name, group in comm_group.items():
        for rank in group["ranks"]:
            pg_for_rank[int(rank)] = str(pg_name)
    npus_count = len(pg_for_rank)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    digests = []
    for rank in range(npus_count):
        path = OUTPUT_DIR / f"{TRACE_PREFIX}.{rank}.et"
        with path.open("wb") as output:
            encode_message(output, build_metadata())

            invalid = ChakraNode()
            invalid.id = 0
            invalid.name = "fixture_invalid"
            invalid.type = 0  # INVALID_NODE
            invalid.attr.append(ChakraAttr(name="is_cpu_op", bool_val=False))

            comp = ChakraNode()
            comp.id = 1
            comp.name = "fixture_comp"
            comp.type = COMP_NODE
            comp.attr.append(uint64_attr("num_ops", 10000))
            comp.attr.append(uint64_attr("tensor_size", 4096))

            coll = ChakraNode()
            coll.id = 2
            coll.name = "fixture_all_reduce"
            coll.type = COMM_COLL_NODE
            coll.attr.append(ChakraAttr(name="comm_type", uint64_val=ALL_REDUCE))
            coll.attr.append(uint64_attr("comm_size", 4096))
            coll.attr.append(ChakraAttr(name="comm_priority", uint32_val=0))
            coll.attr.append(
                ChakraAttr(name="pg_name", string_val=pg_for_rank[rank])
            )
            coll.attr.append(
                ChakraAttr(
                    name="involved_dim", bool_list=BoolList(values=[True, True])
                )
            )

            # The remote-memory FIFO only exists on the 26 boundary-port
            # ranks (edge_remote_memory_pool profile, npu-ids); non-port
            # ranks must not emit a remote MEM node (AnalyticalRemoteMemory
            # fails closed with "no configured remote-memory port").
            REMOTE_PORT_RANKS = [0, 1, 2, 3, 4, 5, 6, 11, 12, 17, 18, 23, 24, 29, 30, 35, 36, 41, 42, 47, 48, 49, 50, 51, 52, 53]
            nodes = [invalid, comp, coll]
            if rank in REMOTE_PORT_RANKS:
                remote_mem = ChakraNode()
                remote_mem.id = 3
                remote_mem.name = "fixture_remote_mem_load"
                remote_mem.type = MEM_LOAD_NODE
                remote_mem.attr.append(uint64_attr("tensor_size", 4096))
                nodes.append(remote_mem)

            hbm_restore = ChakraNode()
            hbm_restore.id = 4
            hbm_restore.name = "fixture_local_hbm_kv_restore"
            hbm_restore.type = MEM_LOAD_NODE
            hbm_restore.attr.append(uint64_attr("tensor_size", 4096))
            hbm_restore.attr.append(
                ChakraAttr(name="is_local_hbm_kv_restore", bool_val=True)
            )

            nodes.append(hbm_restore)
            for node in nodes:
                encode_message(output, node)

        digests.append(hashlib.sha256(path.read_bytes()).hexdigest())
        print(f"[fixture] wrote {path} (pg={pg_for_rank[rank]})")

    total_digest = hashlib.sha256("\n".join(digests).encode("utf-8")).hexdigest()
    print(
        f"[fixture] generated {npus_count} rank files in {OUTPUT_DIR} "
        f"(digest={total_digest}); expected completions = {npus_count * 4 + 26}  # 26 boundary-port ranks emit the remote MEM node"
    )


if __name__ == "__main__":
    main()
