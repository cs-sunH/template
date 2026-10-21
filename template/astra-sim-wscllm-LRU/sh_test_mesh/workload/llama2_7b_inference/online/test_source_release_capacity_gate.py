#!/usr/bin/env python3
"""Real-builder/generator regression for source-release capacity gates.

Usage (after activate.sh):
  python3 online/test_source_release_capacity_gate.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict, deque
from pathlib import Path
import sys
import unittest

WORKLOAD_DIR = Path(os.environ.get(
    "WSC_WORKLOAD_DIR", Path(__file__).resolve().parents[1])).resolve()
sys.path[:0] = [
    str(WORKLOAD_DIR),
    str(WORKLOAD_DIR / "online"),
    str(WORKLOAD_DIR.parents[1]),
]

from generate_trace import (  # noqa: E402
    COMM_RECV_NODE,
    COMM_SEND_NODE,
    COMP_NODE,
)
from generate_wsc_llm_trace import PendingHistoryGate  # noqa: E402
from online.graph_batch_builder import GraphBatchBuilder  # noqa: E402
from session_kv_manager import KVTransfer, KVTransferShard  # noqa: E402


PREFILL_RANKS = (0, 1)
DECODE_RANKS = (5, 6)
LAYERS = 4


def _config():
    return type("Config", (), {
        "npus_count": 16,
        "remote_operand_loads": False,
        "trace_granularity": "request_aggregated",
        "inference_groups": [
            type("Group", (), {"name": "g_prefill", "ranks": PREFILL_RANKS,
                               "pg_name": "tp_prefill"})(),
            type("Group", (), {"name": "g_decode", "ranks": DECODE_RANKS,
                               "pg_name": "tp_decode"})(),
        ],
        "prefill_chunk_size": 128,
        "layers": LAYERS,
        "hidden_size": 64,
        "ffn_size": 128,
        "vocab_size": 256,
        "bytes_per_elem": 2,
        "num_heads": 8,
        "mlp_variant": "gelu",
        "request_queue": [
            type("Request", (), {"session_arrival_time_ns": 5000,
                                  "inter_request_interval_ns": None})(),
            type("Request", (), {"session_arrival_time_ns": 9000,
                                  "inter_request_interval_ns": None})(),
        ],
        "model": type("Model", (), {"layers": LAYERS, "hidden_size": 64,
                                     "bytes_per_elem": 2})(),
        "hardware": type("Hardware", (), {"mesh_rows": 4, "mesh_cols": 4,
                                           "npus_count": 16})(),
    })()


def _plan(request_id="req_evict", session_id="session_evict",
          queue_index=0, turn_index=0):
    return {
        "request_id": request_id,
        "session_id": session_id,
        "turn_index": turn_index,
        "queue_index": queue_index,
        "prefill_instance_index": 0,
        "decode_instance_index": 1,
        "history_action": None,
        "history_source_instance_index": None,
        "history_transfer_bytes": 0,
        "history_recompute_tokens": 0,
        "history_tokens_before": 0,
        "prefill_context_tokens": 64,
        "prefill_length": 64,
        "decode_length": 4,
        "history_transfers": (),
        "history_evictions": (),
        "prefill_evictions": (),
    }


def _store(session_id, source_edge_pairs, *, source_instance=1):
    shards = tuple(
        KVTransferShard(
            source_rank=source,
            target_rank=edge,
            edge_rank=edge,
            bytes=900,
            noc_path=(source, edge),
            layer_start=0,
            layer_end=LAYERS,
        )
        for source, edge in source_edge_pairs
    )
    return KVTransfer(
        kind="remote_store",
        phase="history",
        reason="watermark_admission_session",
        session_id=session_id,
        trigger_request_id="req_evict",
        source_instance_index=source_instance,
        target_instance_index=None,
        total_bytes=sum(shard.bytes for shard in shards),
        shards=shards,
        model_layers=LAYERS,
        layer_start=0,
        layer_end=LAYERS,
        resident_prefix_layers_before=LAYERS,
        resident_prefix_layers_after=0,
    )


def _batch_json(builder):
    batch = builder.batch
    return {
        "nodes": list(batch["nodes"]),
        "parent_edges": list(batch["parent_edges"]),
        "touched_ranks": sorted(batch["_touched_ranks"]),
    }


def _parent_ids(batch, rank, node_id):
    return {
        edge["from"] for edge in batch["parent_edges"]
        if edge["rank"] == rank and edge["to"] == node_id
    }


def _nodes(batches):
    return {
        (node["rank"], node["id"]): node
        for batch in batches for node in batch["nodes"]
    }


def _assert_acyclic_with_comm_matches(testcase, batches):
    """Count Python parent edges and actual tag-matched send->recv edges."""
    nodes = _nodes(batches)
    adjacency = defaultdict(set)
    indegree = {key: 0 for key in nodes}

    def add_edge(parent, child):
        testcase.assertIn(parent, nodes, "edge parent is absent: {}".format(parent))
        testcase.assertIn(child, nodes, "edge child is absent: {}".format(child))
        if child not in adjacency[parent]:
            adjacency[parent].add(child)
            indegree[child] += 1

    for batch in batches:
        for edge in batch["parent_edges"]:
            add_edge((edge["rank"], edge["from"]),
                     (edge["rank"], edge["to"]))

    sends = defaultdict(list)
    recvs = defaultdict(list)
    for key, node in nodes.items():
        if node["type"] == COMM_SEND_NODE:
            comm = node["comm"]
            sends[(comm["src"], comm["dst"], comm["tag"])].append(key)
        elif node["type"] == COMM_RECV_NODE:
            comm = node["comm"]
            recvs[(comm["src"], comm["dst"], comm["tag"])].append(key)
    testcase.assertEqual(set(sends), set(recvs), "unmatched comm send/recv")
    for match_key in sends:
        testcase.assertEqual(len(sends[match_key]), 1, "duplicate comm send tag")
        testcase.assertEqual(len(recvs[match_key]), 1, "duplicate comm recv tag")
        add_edge(sends[match_key][0], recvs[match_key][0])

    ready = deque(key for key, count in indegree.items() if count == 0)
    visited = 0
    while ready:
        key = ready.popleft()
        visited += 1
        for child in adjacency[key]:
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(child)
    testcase.assertEqual(visited, len(nodes), "cycle including comm match edges")


def _build_core_fixture(enable):
    config = _config()
    if enable:
        builder = GraphBatchBuilder(config, source_release_capacity_gates=True)
    else:
        builder = GraphBatchBuilder(config)
    batches = []

    # The real P prefill emits its end marker first. Four remote-store shards
    # then start from that marker; two distinct sessions reuse source rank 5.
    root_plan = _plan()
    builder.begin_batch()
    builder.emit_prefill_batch(root_plan)
    stores = (
        _store("victim_A", ((5, 1), (6, 2))),
        _store("victim_B", ((5, 1),)),
        _store("victim_direct", ((0, 0),), source_instance=0),
    )
    for transfer in stores:
        builder.emit_decode_eviction_actions(
            root_plan, (transfer,), gated_on_prefill_segment_end=True)
    release_tails = ({
        rank: tuple(ids)
        for rank, ids in builder.pending_source_release_tails.items()
    } if enable else {})
    decode_plan = dict(root_plan, decode_evictions=())
    builder.emit_iteration_train({
        "train_id": "batch_train_i1_capacity_reuse",
        "instance_index": 1,
        "joiners": [decode_plan],
        "pass_spans": [(1, 65)],
        "iterations": 1,
        "exit_members": [],
    })
    batches.append(_batch_json(builder))

    # An unrelated P session is a distinct-session capacity reuse consumer.
    builder.begin_batch()
    builder.emit_prefill_batch(_plan(
        request_id="req_distinct_session_reuse",
        session_id="session_distinct_reuser",
        queue_index=1,
    ))
    batches.append(_batch_json(builder))
    return builder, batches, release_tails


def _off_snapshot():
    _, batches, _ = _build_core_fixture(False)
    return json.dumps(batches, ensure_ascii=False, separators=(",", ":"))


class SourceReleaseCapacityGateTest(unittest.TestCase):
    def test_legacy_off_graphbatch_json_matches_frozen_bytes(self):
        digest = hashlib.sha256((_off_snapshot() + "\n").encode("utf-8")).hexdigest()
        self.assertEqual(
            digest,
            "64945939abc1ef398486bf8823f87e1770a8ddae29fc4ad41e63790e46a099d0",
        )

    def test_source_release_metadata_distinguishes_edge_store_from_ack(self):
        builder = GraphBatchBuilder(_config(),
                                    source_release_capacity_gates=True)
        builder.begin_batch()
        direct = builder._emit_plan_transfer(
            _plan(), _store("meta_direct", ((0, 0),), source_instance=0),
            "metadata_probe")
        direct_shard = direct["shards"][0]
        self.assertEqual(direct_shard["source_release_dependency"],
                         "mem_store_completion")
        self.assertEqual(direct_shard["source_release_node_id"],
                         direct_shard["edge_store_node_id"])

        ack = builder._emit_plan_transfer(
            _plan(request_id="req_ack", session_id="meta_ack"),
            _store("meta_ack", ((5, 1),)),
            "metadata_probe")
        ack_shard = ack["shards"][0]
        self.assertEqual(ack_shard["source_release_dependency"],
                         "remote_store_ack_recv")
        self.assertEqual(ack_shard["source_rank"], 5)
        self.assertNotEqual(
            (ack_shard["source_rank"], ack_shard["source_release_node_id"]),
            (ack_shard["edge_rank"], ack_shard["edge_store_node_id"]),
        )
        source_nodes = builder.builders[5].nodes
        source_release = next(
            node for node in source_nodes
            if node["id"] == ack_shard["source_release_node_id"])
        self.assertIn("_ack_from_edge1", source_release["name"])

    def test_gated_p_end_multirank_repeated_stores_and_distinct_session_reuse(self):
        builder, batches, tails = _build_core_fixture(True)
        self.assertEqual(set(tails), {0, 5, 6})
        self.assertEqual(len(tails[5]), 2, "same rank releases must aggregate")
        self.assertEqual(len(tails[6]), 1)
        self.assertEqual(len(tails[0]), 1)

        all_nodes = _nodes(batches)
        names_by_key = {key: node["name"] for key, node in all_nodes.items()}
        for node_id in tails[5]:
            self.assertIn("_ack_from_edge1", names_by_key[(5, node_id)])
        self.assertIn("_edge_store", names_by_key[(0, tails[0][0])])

        # The source ACK is distinct from the edge mem_store tail retained for
        # same-session pool-write -> restore ordering.
        pool_tails = builder.pending_store_tails
        self.assertIn("victim_A", pool_tails)
        self.assertIn("victim_B", pool_tails)
        self.assertNotEqual(tails[5][0], pool_tails["victim_A"][0][1])

        d_nodes = batches[0]["nodes"]
        for rank in (5, 6):
            first_compute = next(
                node for node in d_nodes
                if node["rank"] == rank and node["type"] == COMP_NODE
                and not node.get("is_timer_op")
                and node["compute"].get("op_role") != "control_marker"
            )
            self.assertTrue(set(tails[rank]).issubset(
                _parent_ids(batches[0], rank, first_compute["id"])))

        p_nodes = batches[1]["nodes"]
        first_p_compute = next(
            node for node in p_nodes
            if node["rank"] == 0 and node["type"] == COMP_NODE
            and not node.get("is_timer_op")
            and node["compute"].get("op_role") != "control_marker"
        )
        self.assertIn(tails[0][0], _parent_ids(batches[1], 0,
                                               first_p_compute["id"]))
        self.assertFalse(builder.pending_source_release_tails)

        release_keys = {
            (rank, node_id) for rank, ids in tails.items() for node_id in ids
        }
        for batch in batches:
            for edge in batch["parent_edges"]:
                if (edge["rank"], edge["from"]) not in release_keys:
                    continue
                child = all_nodes[(edge["rank"], edge["to"])]
                is_compute = (
                    child["type"] == COMP_NODE
                    and not child.get("is_timer_op")
                    and child["compute"].get("op_role") != "control_marker"
                )
                is_kv_destination = (
                    child["name"].endswith("target_hbm_write_kv")
                    or child["name"].endswith("recv_kv")
                )
                self.assertTrue(
                    is_compute or is_kv_destination,
                    "release tail joined a non-capacity operation: {}".format(
                        child["name"]))
        _assert_acyclic_with_comm_matches(self, batches)

    def test_remote_restore_waits_for_source_ack_and_preserves_pool_write_order(self):
        builder = GraphBatchBuilder(_config(),
                                    source_release_capacity_gates=True)
        store = _store("victim_restore", ((5, 1),))
        builder.begin_batch()
        plan = _plan()
        builder.emit_decode_eviction_actions(plan, (store,))
        builder.emit_decode_eviction_actions(
            _plan(request_id="req_other_victim", session_id="session_other"),
            (_store("victim_other_session", ((5, 1),)),))
        store_batch = _batch_json(builder)
        release_ids = tuple(builder.pending_source_release_tails[5])
        self.assertEqual(len(release_ids), 2)
        pool_edge, pool_store_id = builder.pending_store_tails["victim_restore"][0]
        self.assertEqual(pool_edge, 1)

        load = KVTransfer(
            kind="remote_load", phase="history", reason="history_remote_restore",
            session_id="victim_restore", trigger_request_id="req_restore",
            source_instance_index=None, target_instance_index=1,
            total_bytes=900,
            shards=(KVTransferShard(
                source_rank=1, target_rank=5, edge_rank=1, bytes=900,
                noc_path=(1, 5), layer_start=0, layer_end=LAYERS),),
            model_layers=LAYERS, layer_start=0, layer_end=LAYERS,
            resident_prefix_layers_before=0,
            resident_prefix_layers_after=LAYERS,
        )
        builder.begin_batch()
        pool_arms = builder._arm_store_tail_dependencies(load, "restore")
        marker = builder._mark()
        builder._emit_plan_transfer(
            _plan(request_id="req_restore", session_id="victim_restore",
                  queue_index=1, turn_index=1),
            load,
            "history_transfer",
            gate=PendingHistoryGate(
                source_instance_index=1,
                timer_gates=(None, None),
                location="remote_memory"),
            extra_arm_ids=pool_arms,
        )
        builder._collect(marker)
        restore_batch = _batch_json(builder)
        restore = next(node for node in restore_batch["nodes"]
                       if node["name"].endswith("target_hbm_write_kv"))
        self.assertTrue(set(release_ids).issubset(
            _parent_ids(restore_batch, 5, restore["id"])))
        edge_request = next(node for node in restore_batch["nodes"]
                            if node["name"].endswith("request_from_rank5"))
        self.assertIn(pool_store_id,
                      _parent_ids(restore_batch, 1, edge_request["id"]))
        self.assertFalse(builder.pending_source_release_tails)
        self.assertNotIn("victim_restore", builder.pending_store_tails)
        _assert_acyclic_with_comm_matches(self, [store_batch, restore_batch])

    def test_noc_migration_waits_at_destination_receive(self):
        builder = GraphBatchBuilder(_config(),
                                    source_release_capacity_gates=True)
        store = _store("victim_migrate", ((5, 1),))
        builder.begin_batch()
        builder.emit_decode_eviction_actions(_plan(), (store,))
        store_batch = _batch_json(builder)
        release_id = builder.pending_source_release_tails[5][0]

        migration = KVTransfer(
            kind="noc_migrate", phase="history", reason="history_migration",
            session_id="other_migrating_session", trigger_request_id="req_migrate",
            source_instance_index=0, target_instance_index=1,
            total_bytes=900,
            shards=(KVTransferShard(
                source_rank=0, target_rank=5, edge_rank=None, bytes=900,
                noc_path=(0, 5), layer_start=0, layer_end=LAYERS),),
            model_layers=LAYERS, layer_start=0, layer_end=LAYERS,
            resident_prefix_layers_before=LAYERS,
            resident_prefix_layers_after=LAYERS,
        )
        builder.begin_batch()
        marker = builder._mark()
        builder._emit_plan_transfer(
            _plan(request_id="req_migrate", session_id="other_migrating_session",
                  queue_index=1, turn_index=1),
            migration, "history_transfer")
        builder._collect(marker)
        migration_batch = _batch_json(builder)
        recv = next(node for node in migration_batch["nodes"]
                    if node["name"].endswith("recv_kv"))
        self.assertIn(release_id, _parent_ids(migration_batch, 5, recv["id"]))
        migration_release_id = builder.pending_source_release_tails[0][0]
        self.assertNotEqual((0, migration_release_id), (5, release_id))
        source_ack = next(node for node in migration_batch["nodes"]
                          if node["rank"] == 0
                          and node["id"] == migration_release_id)
        self.assertIn("_ack_from_rank5", source_ack["name"])
        _assert_acyclic_with_comm_matches(self, [store_batch, migration_batch])

        # The migration source keeps its old KV until that ACK receive.  The
        # next same-rank ordinary compute must wait for the physical release.
        builder.begin_batch()
        builder.emit_prefill_batch(_plan(
            request_id="req_after_migration",
            session_id="session_after_migration",
            queue_index=1,
        ))
        next_p_batch = _batch_json(builder)
        p_compute = next(
            node for node in next_p_batch["nodes"]
            if node["rank"] == 0 and node["type"] == COMP_NODE
            and not node.get("is_timer_op")
            and node["compute"].get("op_role") != "control_marker"
        )
        self.assertIn(migration_release_id,
                      _parent_ids(next_p_batch, 0, p_compute["id"]))
        self.assertFalse(builder.pending_source_release_tails)
        _assert_acyclic_with_comm_matches(
            self, [store_batch, migration_batch, next_p_batch])

    def test_migration_source_ack_gates_following_restore_and_compute(self):
        builder = GraphBatchBuilder(_config(),
                                    source_release_capacity_gates=True)
        migration = KVTransfer(
            kind="noc_migrate", phase="history", reason="history_migration",
            session_id="session_migrating", trigger_request_id="req_migrate",
            source_instance_index=0, target_instance_index=1,
            total_bytes=1800,
            shards=(
                KVTransferShard(
                    source_rank=0, target_rank=5, edge_rank=None, bytes=900,
                    noc_path=(0, 5), layer_start=0, layer_end=LAYERS),
                KVTransferShard(
                    source_rank=1, target_rank=6, edge_rank=None, bytes=900,
                    noc_path=(1, 6), layer_start=0, layer_end=LAYERS),
            ),
            model_layers=LAYERS, layer_start=0, layer_end=LAYERS,
            resident_prefix_layers_before=LAYERS,
            resident_prefix_layers_after=LAYERS,
        )
        builder.begin_batch()
        marker = builder._mark()
        migration_record = builder._emit_plan_transfer(
            _plan(request_id="req_migrate", session_id="session_migrating",
                  turn_index=1),
            migration, "history_transfer")
        builder._collect(marker)
        migration_batch = _batch_json(builder)
        release_ids = {
            shard["source_rank"]: shard["source_release_node_id"]
            for shard in migration_record["shards"]
        }
        self.assertEqual(set(release_ids), {0, 1})
        for rank, node_id in release_ids.items():
            ack = next(node for node in migration_batch["nodes"]
                       if node["rank"] == rank and node["id"] == node_id)
            self.assertIn("_ack_from_rank", ack["name"])

        # The old source rank 0 is also the next restore destination.  Its
        # HBM write must wait for the migration ACK before allocating KV.
        remote_load = KVTransfer(
            kind="remote_load", phase="history", reason="history_restore",
            session_id="session_migrating", trigger_request_id="req_restore",
            source_instance_index=None, target_instance_index=0,
            total_bytes=900,
            shards=(KVTransferShard(
                source_rank=1, target_rank=0, edge_rank=1, bytes=900,
                noc_path=(1, 0), layer_start=0, layer_end=LAYERS),),
            model_layers=LAYERS, layer_start=0, layer_end=LAYERS,
            resident_prefix_layers_before=0,
            resident_prefix_layers_after=LAYERS,
        )
        builder.begin_batch()
        marker = builder._mark()
        builder._emit_plan_transfer(
            _plan(request_id="req_restore", session_id="session_migrating",
                  queue_index=1, turn_index=1),
            remote_load, "history_transfer",
            gate=PendingHistoryGate(
                source_instance_index=0,
                timer_gates=(None, None),
                location="remote_memory"),
        )
        builder._collect(marker)
        restore_batch = _batch_json(builder)
        restore = next(node for node in restore_batch["nodes"]
                       if node["name"].endswith("target_hbm_write_kv"))
        self.assertIn(release_ids[0],
                      _parent_ids(restore_batch, 0, restore["id"]))
        self.assertEqual(builder.pending_source_release_tails,
                         {1: [release_ids[1]]})

        # The other migration ACK remains live across batches until rank 1's
        # next normal compute consumes it.
        builder.begin_batch()
        builder.emit_prefill_batch(_plan(
            request_id="req_after_migration",
            session_id="session_after_migration",
            queue_index=1,
        ))
        compute_batch = _batch_json(builder)
        p_compute = next(
            node for node in compute_batch["nodes"]
            if node["rank"] == 1 and node["type"] == COMP_NODE
            and not node.get("is_timer_op")
            and node["compute"].get("op_role") != "control_marker"
        )
        self.assertIn(release_ids[1],
                      _parent_ids(compute_batch, 1, p_compute["id"]))
        self.assertFalse(builder.pending_source_release_tails)
        _assert_acyclic_with_comm_matches(
            self, [migration_batch, restore_batch, compute_batch])

    def test_legacy_mode_has_no_release_edges(self):
        _, batches, _ = _build_core_fixture(False)
        ack_ids = {
            (node["rank"], node["id"])
            for batch in batches for node in batch["nodes"]
            if "_ack_from_edge" in node["name"]
        }
        self.assertTrue(all(
            (edge["rank"], edge["from"]) not in ack_ids
            for batch in batches for edge in batch["parent_edges"]
        ))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump-off", action="store_true")
    args, rest = parser.parse_known_args()
    if args.dump_off:
        print(_off_snapshot())
        return
    unittest.main(argv=[sys.argv[0], *rest])


if __name__ == "__main__":
    main()
