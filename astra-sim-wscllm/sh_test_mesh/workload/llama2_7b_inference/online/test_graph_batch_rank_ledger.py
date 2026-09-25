#!/usr/bin/env python3
"""Focused regression for exact GraphBatch touched-rank accounting."""
import os
import sys
from types import SimpleNamespace
from unittest import mock

_ONLINE_DIR = os.path.dirname(os.path.abspath(__file__))
_WORKLOAD_DIR = os.path.dirname(_ONLINE_DIR)
for _path in (_ONLINE_DIR, _WORKLOAD_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from online import online_scheduler_base as scheduler_base_module  # noqa: E402
from online.graph_batch_builder import GraphBatchBuilder  # noqa: E402
from online.online_scheduler_base import OnlineSchedulerBase  # noqa: E402


class _NoNodeIteration(list):
    """Fails if rank bookkeeping makes an avoidable full node traversal."""

    def __init__(self):
        super().__init__([{"rank": 99}])
        self.iteration_count = 0

    def __iter__(self):
        self.iteration_count += 1
        raise AssertionError("unexpected full GraphBatch node traversal")


def _base_for(graph_batch):
    scheduler = OnlineSchedulerBase.__new__(OnlineSchedulerBase)
    scheduler.graph = SimpleNamespace(batch=graph_batch)
    scheduler._batch = {
        "delivery_sequence": 7,
        "tick": 123,
        "reasons": ["rank-ledger"],
        "watches": [],
        "assignments": [],
        "kv_actions": [],
        "future_alarms": [],
    }
    scheduler._provisional_kv_actions = {}
    scheduler.digest_sink = None
    scheduler._profile_batch = {"scanned_entries": 0, "full_scan_entries": 0}
    scheduler.profile_sink = None
    return scheduler


def test_collect_tracks_only_ranks_that_emitted_nodes():
    """The private ledger is exact and records no edge-only/idle rank."""
    config = SimpleNamespace(
        npus_count=3,
        remote_operand_loads=False,
        inference_groups=(),
        prefill_chunk_size=1,
    )
    builder = GraphBatchBuilder(config)
    builder.begin_batch()
    marker = builder._mark()
    builder.builders[0].comp("rank_0", 1, 1)
    builder.builders[2].comp("rank_2", 1, 1)
    builder._collect(marker)

    assert builder.batch["_touched_ranks"] == {0, 2}
    assert [node["rank"] for node in builder.batch["nodes"]] == [0, 2]


def test_build_reuses_collected_rank_ledger_without_node_scan():
    """Normal (digest-off) GraphBatch assembly never traverses nodes for ranks."""
    nodes = _NoNodeIteration()
    scheduler = _base_for({
        "nodes": nodes,
        "parent_edges": [],
        "_touched_ranks": {2, 0},
    })

    batch = scheduler.build_graph_batch()

    assert batch["touched_ranks"] == [0, 2]
    assert nodes.iteration_count == 0


def test_digest_reuses_public_rank_metadata_without_second_node_scan():
    """Digest serialization is the sole intended node walk when enabled."""
    nodes = _NoNodeIteration()
    scheduler = OnlineSchedulerBase.__new__(OnlineSchedulerBase)
    scheduler._batch = {"tick": 123, "reasons": ["rank-ledger"]}
    batch = {
        "batch_id": 7,
        "nodes": nodes,
        "parent_edges": [],
        "watches": [],
        "future_alarms": [],
        "touched_ranks": [0, 2],
    }

    # Stub the one required JSON serialization to expose any *additional*
    # iteration that rank calculation would otherwise introduce.
    with mock.patch.object(scheduler_base_module.json, "dumps",
                           return_value="payload"):
        digest = scheduler._digest_row(batch)

    assert digest["ranks"] == [0, 2]
    assert nodes.iteration_count == 0
