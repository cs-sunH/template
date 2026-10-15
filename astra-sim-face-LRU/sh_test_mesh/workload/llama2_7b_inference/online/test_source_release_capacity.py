"""Real NOC data/ACK dependencies across batches and source capacity reuse."""
from collections import defaultdict, deque
import copy
import unittest

import generate_face_trace as trace
from generate_trace import COMM_SEND_NODE, COMM_RECV_NODE
from online.graph_batch_builder import GraphBatchBuilder
from online.test_graph_batch_builder import _make_config


def batch_nodes(graph):
    return {(node["rank"], node["id"]): node for node in graph.batch["nodes"]}


def parents(graph, rank, node_id):
    return {edge["from"] for edge in graph.batch["parent_edges"]
            if edge["rank"] == rank and edge["to"] == node_id}


def assert_acyclic(test, nodes, edges):
    successors = defaultdict(set)
    incoming = {node: 0 for node in nodes}
    def add(source, target):
        test.assertIn(source, nodes)
        test.assertIn(target, nodes)
        if target not in successors[source]:
            successors[source].add(target)
            incoming[target] += 1
    for edge in edges:
        add((edge["rank"], edge["from"]), (edge["rank"], edge["to"]))
    sends, recvs = defaultdict(list), defaultdict(list)
    for key, node in nodes.items():
        if node["type"] in (COMM_SEND_NODE, COMM_RECV_NODE):
            comm = node["comm"]
            (sends if node["type"] == COMM_SEND_NODE else recvs)[
                (comm["src"], comm["dst"], comm["tag"])].append(key)
    test.assertEqual(set(sends), set(recvs))
    for key in sends:
        test.assertEqual(len(sends[key]), 1, "ambiguous tag pairing")
        test.assertEqual(len(recvs[key]), 1, "ambiguous tag pairing")
        add(sends[key][0], recvs[key][0])
    ready = deque(key for key, degree in incoming.items() if degree == 0)
    visited = 0
    while ready:
        key = ready.popleft()
        visited += 1
        for target in successors[key]:
            incoming[target] -= 1
            if incoming[target] == 0:
                ready.append(target)
    test.assertEqual(visited, len(nodes), "dependency cycle including actual communication")


def transfer(graph, source=0, target=1, queue_index=0, category=3000):
    routes = trace._paired_transfer(config=graph.config, builders=graph.builders,
        queue_index=queue_index, category=category, name=f"move_{queue_index}_{category}",
        source_group=graph.group_by_index[source], target_group=graph.group_by_index[target],
        total_bytes=256, source_release_ack=graph.source_release_capacity_gates,
        source_release_ack_tag_factory=graph._source_release_ack_tag,
        source_release_arm_ids=graph._take_source_release_arms(graph.group_by_index[target].ranks))
    graph._register_source_release_routes(routes)
    return routes


class SourceReleaseCapacityTests(unittest.TestCase):
    def test_default_off_preserves_graph_and_has_no_ack(self):
        for invalid in (None, 0, 1, "true"):
            with self.assertRaises(TypeError):
                GraphBatchBuilder(_make_config(), source_release_capacity_gates=invalid)
        outputs = []
        for explicit in (False, True):
            graph = (GraphBatchBuilder(_make_config(), source_release_capacity_gates=False)
                     if explicit else GraphBatchBuilder(_make_config()))
            graph.begin_batch()
            mark = graph._mark()
            routes = transfer(graph)
            graph._collect(mark)
            outputs.append(copy.deepcopy(graph.batch))
            self.assertFalse(any("ack_tag" in route for route in routes))
            self.assertEqual(graph.pending_source_release_tails, {})
        self.assertEqual(outputs[0], outputs[1])

    def test_real_ack_source_reuse_survives_payload_collection_and_marker(self):
        graph = GraphBatchBuilder(_make_config(), source_release_capacity_gates=True)
        graph.begin_batch()
        mark = graph._mark()
        routes = transfer(graph)
        graph._collect(mark)
        previous_nodes = batch_nodes(graph)
        previous_edges = copy.deepcopy(graph.batch["parent_edges"])
        pending = copy.deepcopy(graph.pending_source_release_tails)
        for route in routes:
            source, target = route["source_rank"], route["target_rank"]
            release = (source, route["source_release_node_id"])
            ack = previous_nodes[release]
            self.assertEqual(ack["type"], COMM_RECV_NODE)
            self.assertEqual(ack["comm"]["bytes"], 1)
            self.assertEqual(ack["comm"]["tag"], route["ack_tag"])
            ack_send = next(node for node in previous_nodes.values()
                if node["type"] == COMM_SEND_NODE and node["rank"] == target
                and node["comm"]["tag"] == route["ack_tag"])
            data_recv = next(node for node in previous_nodes.values()
                if node["type"] == COMM_RECV_NODE and node["rank"] == target
                and node["comm"]["bytes"] > 1)
            self.assertIn(data_recv["id"], parents(graph, target, ack_send["id"]))
        graph.begin_batch()
        self.assertEqual(graph.pending_source_release_tails, pending)
        mark = graph._mark()
        for rank in graph.group_by_index[0].ranks:
            graph.builders[rank].comp("control", 1, 0, op_role="control_marker")
        self.assertEqual(graph.pending_source_release_tails, pending)
        for rank in graph.group_by_index[0].ranks:
            graph.builders[rank].comp("new_capacity_user", 1, 64)
        graph._collect(mark)
        self.assertFalse(graph.pending_source_release_tails)
        for route in routes:
            rank = route["source_rank"]
            comp = next(node for node in graph.batch["nodes"] if node["rank"] == rank
                        and node["name"] == "new_capacity_user")
            self.assertIn(route["source_release_node_id"], parents(graph, rank, comp["id"]))
        assert_acyclic(self, previous_nodes | batch_nodes(graph),
                       previous_edges + graph.batch["parent_edges"])

    def test_ack_namespace_is_disjoint_at_large_queue_indices(self):
        config = _make_config()
        config.request_queue = config.request_queue * 2000
        graph = GraphBatchBuilder(config, source_release_capacity_gates=True)
        graph.begin_batch()
        mark = graph._mark()
        routes = transfer(graph, queue_index=1999)
        graph._collect(mark)
        for route in routes:
            self.assertGreater(route["ack_tag"], len(config.request_queue) * 10000)
        assert_acyclic(self, batch_nodes(graph), graph.batch["parent_edges"])

    def test_target_copy_waits_for_all_source_releases_and_tags_do_not_collide(self):
        graph = GraphBatchBuilder(_make_config(), source_release_capacity_gates=True)
        graph.begin_batch()
        mark = graph._mark()
        old = transfer(graph, queue_index=2, category=1000)
        graph._collect(mark)
        nodes, edges = batch_nodes(graph), copy.deepcopy(graph.batch["parent_edges"])
        graph.begin_batch()
        mark = graph._mark()
        new = transfer(graph, source=1, target=0, queue_index=3, category=3000)
        graph._collect(mark)
        for route in old:
            rank = route["source_rank"]
            recv = next(node for node in graph.batch["nodes"] if node["rank"] == rank
                        and node["type"] == COMM_RECV_NODE and node["comm"]["bytes"] > 1)
            self.assertIn(route["source_release_node_id"], parents(graph, rank, recv["id"]))
        self.assertTrue({r["ack_tag"] for r in old}.isdisjoint({r["ack_tag"] for r in new}))
        assert_acyclic(self, nodes | batch_nodes(graph), edges + graph.batch["parent_edges"])


class RemoteSourceReleaseCapacityTests(unittest.TestCase):
    def test_all_cold_source_releases_gate_compute_and_unrelated_restore(self):
        from online import test_store_restore_ordering as fixture
        from dataclasses import replace
        for consumer in ("compute", "restore"):
            with self.subTest(consumer=consumer):
                graph = GraphBatchBuilder(fixture._make_config(), source_release_capacity_gates=True)
                graph.begin_batch()
                mark = graph._mark()
                source_tails = defaultdict(list)
                for index in range(2):
                    plan = {"request_id": "evict", "session_id": "trigger", "turn_index": 0, "queue_index": 0}
                    transfer_ = replace(fixture._store_transfer(), session_id=f"cold_{index}")
                    graph._set_context(plan, "prefill", 0)
                    record = graph._emit_plan_transfer(plan, transfer_, "evict")
                    graph._register_store_tails(record)
                    for shard in record["shards"]:
                        tail = (shard["edge_store_node_id"] if shard["direct_edge_access"]
                                else shard["source_ack_node_id"])
                        source_tails[shard["source_rank"]].append(tail)
                graph._collect(mark)
                previous_nodes, previous_edges = batch_nodes(graph), copy.deepcopy(graph.batch["parent_edges"])
                self.assertEqual({rank: len(tails) for rank, tails in graph.pending_source_release_tails.items()},
                                 {0: 2, 4: 2})
                graph.begin_batch()
                mark = graph._mark()
                if consumer == "compute":
                    for rank in (0, 4):
                        graph.builders[rank].comp("new_capacity_user", 1, 64)
                else:
                    plan = {"request_id": "restore_other", "session_id": "other", "turn_index": 1, "queue_index": 1}
                    load = fixture._load_transfer(fixture._load_shards({0: 0, 4: 1}), session_id="other")
                    graph._set_context(plan, "prefill", 0)
                    graph._emit_plan_transfer(plan, load, "history_kv_transfer_action",
                        gate=trace.PendingHistoryGate(0, (None, None), "remote_memory"))
                graph._collect(mark)
                self.assertEqual(graph.pending_source_release_tails, {})
                self.assertEqual(set(graph.pending_store_tails), {"cold_0", "cold_1"})
                for rank in (0, 4):
                    capacity_user = next(node for node in graph.batch["nodes"]
                        if node["rank"] == rank and (node["name"] == "new_capacity_user"
                            if consumer == "compute" else node.get("is_local_hbm_kv_restore", False)))
                    self.assertTrue(set(source_tails[rank]).issubset(parents(graph, rank, capacity_user["id"])))
                assert_acyclic(self, previous_nodes | batch_nodes(graph),
                    previous_edges + graph.batch["parent_edges"])


if __name__ == "__main__":
    unittest.main()
