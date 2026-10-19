#!/usr/bin/env python3
"""make_hbm_nway_fixture.py -- N-way HBM contention fixture generator.

Writes the self-contained 2-rank fixture inputs into
  sh_test_mesh/generated/hbm_nway_fixture/
  fixture.{0,1}.et      dependency-free per-rank graphs (see below)
  system_on.json        hbm-bandwidth-contention = 1 (N-way model owns
                        COMP / restore / comm / pool endpoints)
  system_off.json       hbm-bandwidth-contention = 0 (legacy behavior:
                        only the two-user COMP+restore 50/50 sharing of
                        hbm-kv-restore-bandwidth-sharing stays)
  network.yml           2-rank Line topology (d2d 4050 B/ns-ish fluid links)
  remote_memory.json    PER_NPU ports on both ranks (6 B/ns, 100 ns)
  comm_group.json       empty (no collectives in this fixture)

Fixture graph (dependency-free nodes, so every slot-compatible node issues
at tick 0; ids ascending = issue order):

  rank 0
    0 COMP_NODE        num_ops=1 tensor_size=300  -> COMP job 300 B
    1 MEM_LOAD restore tensor_size=600            -> RESTORE job 600 B
    2 COMM_SEND -> r1 comm_size=900 tag=1         -> COMM_READ 900 B (join
                          with the network PacketSent)
    3 COMM_SEND -> r1 comm_size=64  tag=2 hbm-charge=false
                                                -> NO HBM job (pass-through)
    4 COMP_NODE        num_ops=1 tensor_size=300 duration_micros=5
                                        -> calibrated: NO HBM job; the
                                           single COMP slot serializes it
                                           after node 0, terminal = issue
                                           + 5000 (ON 5400 / OFF 5300)
                                           (中-5 honor pin)
  rank 1
    0 COMP_NODE        num_ops=1 tensor_size=300  -> COMP job 300 B
    1 MEM_LOAD restore tensor_size=600            -> RESTORE job 600 B
    2 COMM_RECV <- r0  comm_size=900 tag=1        -> COMM_WRITE 900 B (join
                          with the network PacketReceived)
    3 MEM_STORE tensor_size=1200 hbm-access-mode=1 -> POOL_READ 1200 B (join
                          with the remote-memory port transaction)

Analytic expectations with local-mem-bw = 3 GB/s (= 3 B/ns) and
local-mem-latency = 100 ns are asserted by hbm_nway_test.cc:

  contention ON, rank 0 (COMP 300 / RESTORE 600 / COMM_READ 900):
    3-way equal split @ 1 B/ns each from t=100 -> COMP done t=400;
    2-way @ 1.5 -> RESTORE done t=600; solo @ 3 -> COMM_READ done t=700;
    send(2) terminal = join(net ~15, hbm 700) = 700; send(3) has no job
    and issues at tick 0 alongside send(2) (comm slot is an UNLIMITED
    counted slot since 2026-09-25; network-only terminal = 6)
    (comm_read bytes stay exactly 900, not 964); peak jobs = 3;
    redistributions = 4 (2 joining issues + 2 completions with survivors).
  contention ON, rank 1 (COMP 300 / RESTORE 600 / COMM_WRITE 900 /
    POOL_READ 1200): 4-way @ 0.75 -> COMP t=500; 3-way @ 1 -> RESTORE
    t=800; 2-way @ 1.5 -> COMM_WRITE t=1000; solo @ 3 -> POOL_READ
    t=1100; MEM_STORE terminal = join(port t=300, hbm t=1100) = 1100;
    peak jobs = 4; redistributions = 6.
  contention OFF (both ranks): only COMP+RESTORE share 50/50:
    COMP t=300, RESTORE t=400; comm/pool served bytes = 0; peak jobs = 2.
  both configs: the calibrated COMP never enters the model (peak jobs /
    served bytes / all model timings unchanged; its terminal = issue tick
    + 5000: ON 5400 / OFF 5300).

The generated/ directory is gitignored by design; rerun this script to
regenerate the fixture.
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
    COMM_RECV_NODE,
    COMM_SEND_NODE,
    COMP_NODE,
    MEM_LOAD_NODE,
    MEM_STORE_NODE,
    AttributeProto as ChakraAttr,
    GlobalMetadata,
    Node as ChakraNode,
)
from extern.graph_frontend.chakra.src.third_party.utils.protolib import (  # noqa: E402
    encodeMessage as encode_message,
)

OUTPUT_DIR = PROJECT_ROOT / "sh_test_mesh" / "generated" / "hbm_nway_fixture"
TRACE_PREFIX = "fixture"

# Fixture physics (units: a GB/s value is also bytes/ns after the engine's
# *1e9/1e9 cancellation -- local-mem-bw 3 GB/s == 3 B/ns full rate; the
# test's analytic expectations use B/ns).
LOCAL_MEM_BW_GBPS = 3            # 3 B/ns full rate
LOCAL_MEM_LATENCY_NS = 100
PEAK_PERF_TFLOPS = 1000
REMOTE_MEM_BW_GBPS = 6           # 6 B/ns per pool port
REMOTE_MEM_LATENCY_NS = 100


def uint64_attr(name: str, value: int) -> ChakraAttr:
    return ChakraAttr(name=name, uint64_val=max(1, int(value)))


def build_metadata() -> GlobalMetadata:
    metadata = GlobalMetadata(version="0.0.4")
    metadata.attr.extend(
        [
            ChakraAttr(name="schema", string_val="1.0.2-chakra.0.0.4"),
            ChakraAttr(name="execution_mode", string_val="hbm_nway_fixture"),
            ChakraAttr(name="trace_granularity", string_val="synthetic"),
        ]
    )
    return metadata


def rank0_nodes() -> list[ChakraNode]:
    nodes = []

    comp = ChakraNode()
    comp.id = 0
    comp.name = "r0_comp"
    comp.type = COMP_NODE
    comp.attr.append(uint64_attr("num_ops", 1))
    comp.attr.append(uint64_attr("tensor_size", 300))
    nodes.append(comp)

    restore = ChakraNode()
    restore.id = 1
    restore.name = "r0_kv_restore"
    restore.type = MEM_LOAD_NODE
    restore.attr.append(uint64_attr("tensor_size", 600))
    restore.attr.append(
        ChakraAttr(name="is_local_hbm_kv_restore", bool_val=True)
    )
    nodes.append(restore)

    charged_send = ChakraNode()
    charged_send.id = 2
    charged_send.name = "r0_send_charged"
    charged_send.type = COMM_SEND_NODE
    charged_send.attr.extend([
        ChakraAttr(name="comm_src", uint32_val=0),
        ChakraAttr(name="comm_dst", uint32_val=1),
        uint64_attr("comm_size", 900),
        ChakraAttr(name="comm_tag", uint32_val=1),
    ])
    nodes.append(charged_send)

    passthrough_send = ChakraNode()
    passthrough_send.id = 3
    passthrough_send.name = "r0_send_passthrough"
    passthrough_send.type = COMM_SEND_NODE
    passthrough_send.attr.extend([
        ChakraAttr(name="comm_src", uint32_val=0),
        ChakraAttr(name="comm_dst", uint32_val=1),
        uint64_attr("comm_size", 64),
        ChakraAttr(name="comm_tag", uint32_val=2),
        ChakraAttr(name="hbm-charge", bool_val=False),
    ])
    nodes.append(passthrough_send)

    calibrated_comp = ChakraNode()
    calibrated_comp.id = 4
    calibrated_comp.name = "r0_comp_calibrated"
    calibrated_comp.type = COMP_NODE
    calibrated_comp.attr.append(uint64_attr("num_ops", 1))
    calibrated_comp.attr.append(uint64_attr("tensor_size", 300))
    # 中-5 regression: calibrated COMP (duration_micros=5 -> runtime_ns=5000).
    # Honor contract: bypasses the fluid model; the static single COMP slot
    # serializes it after node 0, so its terminal = issue tick + 5000
    # (ON: 400+5000=5400; OFF: 300+5000=5300).
    calibrated_comp.duration_micros = 5
    nodes.append(calibrated_comp)

    return nodes


def rank1_nodes() -> list[ChakraNode]:
    nodes = []

    comp = ChakraNode()
    comp.id = 0
    comp.name = "r1_comp"
    comp.type = COMP_NODE
    comp.attr.append(uint64_attr("num_ops", 1))
    comp.attr.append(uint64_attr("tensor_size", 300))
    nodes.append(comp)

    restore = ChakraNode()
    restore.id = 1
    restore.name = "r1_kv_restore"
    restore.type = MEM_LOAD_NODE
    restore.attr.append(uint64_attr("tensor_size", 600))
    restore.attr.append(
        ChakraAttr(name="is_local_hbm_kv_restore", bool_val=True)
    )
    nodes.append(restore)

    charged_recv = ChakraNode()
    charged_recv.id = 2
    charged_recv.name = "r1_recv_charged"
    charged_recv.type = COMM_RECV_NODE
    charged_recv.attr.extend([
        ChakraAttr(name="comm_src", uint32_val=0),
        ChakraAttr(name="comm_dst", uint32_val=1),
        uint64_attr("comm_size", 900),
        ChakraAttr(name="comm_tag", uint32_val=1),
    ])
    nodes.append(charged_recv)

    pool_store = ChakraNode()
    pool_store.id = 3
    pool_store.name = "r1_pool_store_read"
    pool_store.type = MEM_STORE_NODE
    pool_store.attr.append(uint64_attr("tensor_size", 1200))
    pool_store.attr.append(uint64_attr("hbm-access-mode", 1))
    nodes.append(pool_store)

    return nodes


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
        "roofline-enabled": 1,
        "hbm-kv-restore-bandwidth-sharing": 1,
        "replay-only": 0,
        "track-local-mem": 0,
        "trace-enabled": 0,
        "peak-perf": PEAK_PERF_TFLOPS,
        "local-mem-bw": LOCAL_MEM_BW_GBPS,
        "local-mem-latency": LOCAL_MEM_LATENCY_NS,
        "remote-mem-bw": REMOTE_MEM_BW_GBPS,
        "remote-mem-latency": REMOTE_MEM_LATENCY_NS,
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    digests = []
    for rank, builder in ((0, rank0_nodes), (1, rank1_nodes)):
        path = OUTPUT_DIR / f"{TRACE_PREFIX}.{rank}.et"
        with path.open("wb") as output:
            encode_message(output, build_metadata())
            for node in builder():
                encode_message(output, node)
        digests.append(hashlib.sha256(path.read_bytes()).hexdigest())
        print(f"[fixture] wrote {path}")

    system_on = base_system()
    system_on["hbm-bandwidth-contention"] = 1
    system_off = base_system()
    system_off["hbm-bandwidth-contention"] = 0
    (OUTPUT_DIR / "system_on.json").write_text(
        json.dumps(system_on, indent=2) + "\n", encoding="utf-8"
    )
    (OUTPUT_DIR / "system_off.json").write_text(
        json.dumps(system_off, indent=2) + "\n", encoding="utf-8"
    )

    (OUTPUT_DIR / "network.yml").write_text(
        "# hbm_nway_fixture synthetic 2-rank Line topology (2 dims: the\n"
        "# system template carries per-dimension collective impls)\n"
        "topology: [ Line, Line ]\n"
        "npus_count: [ 2, 1 ]\n"
        "bandwidth: [ 4050.0, 4050.0 ]\n"
        "latency: [ 5, 5 ]\n",
        encoding="utf-8",
    )
    (OUTPUT_DIR / "remote_memory.json").write_text(
        json.dumps(
            {
                "memory-type": "PER_NPU_MEMORY_EXPANSION",
                "remote-mem-bw": REMOTE_MEM_BW_GBPS,
                "remote-mem-latency": REMOTE_MEM_LATENCY_NS,
                "logical-pool": "hbm-nway-fixture-pool",
                "npu-ids": [0, 1],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (OUTPUT_DIR / "comm_group.json").write_text(
        "{}\n", encoding="utf-8"
    )

    total_digest = hashlib.sha256("\n".join(digests).encode("utf-8")).hexdigest()
    print(
        f"[fixture] generated 2 rank files + configs in {OUTPUT_DIR} "
        f"(digest={total_digest}); run hbm_nway_test twice, once with "
        f"system_on.json and once with system_off.json"
    )


if __name__ == "__main__":
    main()
