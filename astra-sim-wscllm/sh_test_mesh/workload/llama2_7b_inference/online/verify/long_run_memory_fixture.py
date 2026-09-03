#!/usr/bin/env python3
"""Long-run ownership regression for the online scheduler.

The default drives one million decision deliveries.  It deliberately uses a
lazy manifest so the fixture itself cannot hide scheduler retention behind a
million resident request dictionaries.  Every response is acknowledged before
the next delivery, matching the production single-in-flight bridge contract.

Set SCHEDULER_MEMORY_DELIVERIES to a smaller value for local smoke runs (the
value includes the final completion-only delivery and must be at least two).
"""

import os
import sys


_ONLINE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_WORKLOAD_DIR = os.path.dirname(_ONLINE_DIR)
for _path in (_ONLINE_DIR, _WORKLOAD_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from online.graph_batch_builder import GraphBatchBuilder  # noqa: E402
from online.online_scheduler_base import OnlineSchedulerBase  # noqa: E402


class _LazyRequests:
    """A sized iterable whose records are created only while base initializes."""

    def __init__(self, count: int):
        self._count = count

    def __len__(self) -> int:
        return self._count

    def __iter__(self):
        for index in range(self._count):
            yield {"request_id": _request_id(index)}


class _FixtureConfig:
    """Only the GraphBatchBuilder construction contract is needed here."""

    npus_count = 1
    remote_operand_loads = False
    inference_groups = ()
    prefill_chunk_size = 1


class _MemoryScheduler(OnlineSchedulerBase):
    """Minimal policy that still creates and collects a real graph node."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.graph = GraphBatchBuilder(self.config)

    def run_variant_policy(self, delta: dict) -> None:
        marker = self.graph._mark()
        builder = self.graph.builders[0]
        request_id = (delta["arrivals"][0]["request_id"]
                      if delta["arrivals"]
                      else delta["completed_groups"][0]["request_id"])
        builder.set_context(request_id,
                            "memory_fixture", 0)
        builder.comp("memory_fixture_node", 1, 1)
        self.graph._collect(marker)


_RETIRED_CONTAINER_NAMES = (
    "manifest",
    "request_by_id",
    "completed_request_ids",
    "_seen_ack_delivery_seqs",
    "injected_unfinished_history",
    "ledger_completed_unreconciled",
)


def _discard_row(_row: dict) -> None:
    """Prevent optional audit rows from becoming fixture-side history."""


def _request_id(index: int) -> str:
    return "m{}".format(index)


def _positive_env(name: str, default: int, *, minimum: int = 1) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise RuntimeError("{} must be an integer, got {!r}".format(name, raw)) from exc
    if value < minimum:
        raise RuntimeError("{} must be >= {}, got {}".format(name, minimum, value))
    return value


def _rss_mib():
    """Best-effort diagnostic only; structural checks are the pass criterion."""

    try:
        with open("/proc/self/statm", "r", encoding="utf-8") as source:
            resident_pages = int(source.read().split()[1])
        return resident_pages * os.sysconf("SC_PAGE_SIZE") / (1024 * 1024)
    except (IndexError, OSError, ValueError):
        return None


def _assert_no_retired_containers(scheduler) -> None:
    retained = [
        name for name in _RETIRED_CONTAINER_NAMES if hasattr(scheduler, name)
    ]
    if retained:
        raise RuntimeError("retired long-run containers still exist: {}".format(
            ", ".join(retained)))


def _assert_after_ack(scheduler, seq: int) -> None:
    if scheduler._delivery_reply_cache is not None:
        raise RuntimeError("delivery {} retained its reply cache after ack".format(seq))
    if scheduler._batch is not None:
        raise RuntimeError("delivery {} retained its scheduler batch after ack".format(seq))
    if scheduler.graph.batch is not None:
        raise RuntimeError("delivery {} retained its graph batch after ack".format(seq))
    if getattr(scheduler.graph, "batch_first_step", False):
        raise RuntimeError("delivery {} retained graph first-step state after ack".format(seq))
    for rank, builder in scheduler.graph.builders.items():
        if builder.nodes or builder.edges:
            raise RuntimeError(
                "delivery {} left graph-builder tails on rank {} "
                "(nodes={} edges={})".format(
                    seq, rank, len(builder.nodes), len(builder.edges)))
    if scheduler._emitted_by_delivery:
        raise RuntimeError("delivery {} retained emitted-record state after ack".format(seq))
    if scheduler._acked_through != seq or scheduler._acked_out_of_order:
        raise RuntimeError(
            "delivery {} did not compact ack state: through={} holes={}".format(
                seq, scheduler._acked_through,
                sorted(scheduler._acked_out_of_order)))


def _assert_lifecycle(scheduler, seq: int, request_count: int) -> None:
    expected_arrived = min(seq + 1, request_count)
    expected_completed = min(seq, request_count)
    if len(scheduler.in_flight) > 1:
        raise RuntimeError("delivery {} has more than one in-flight request".format(seq))
    if scheduler.arrived_request_count != expected_arrived:
        raise RuntimeError("arrival counter diverged at delivery {}".format(seq))
    if scheduler.completed_request_count != expected_completed:
        raise RuntimeError("completion counter diverged at delivery {}".format(seq))
    if len(scheduler.unseen_request_ids) != request_count - expected_arrived:
        raise RuntimeError("unseen request index did not shrink at delivery {}".format(seq))
    if scheduler.delivery_count != seq + 1 or scheduler.ack_count != seq + 1:
        raise RuntimeError("delivery/ack counters diverged at delivery {}".format(seq))


def _delta(seq: int, request_count: int) -> dict:
    arrival_index = seq if seq < request_count else None
    completed_index = seq - 1 if seq > 0 else None
    return {
        "schema_version": 1,
        "delivery_sequence": seq,
        "delivery_epoch": seq,
        "tick": seq,
        "deferred_from_tick": seq,
        "reasons": ["MEMORY_FIXTURE"],
        "arrivals": ([] if arrival_index is None else [{
            "request_id": _request_id(arrival_index),
            "queue_index": arrival_index,
            "ingress_seq": arrival_index,
        }]),
        "completed_groups": ([] if completed_index is None else [{
            "request_id": _request_id(completed_index),
            "stage": "",
        }]),
        "completed_nodes": [],
        "retry_items": [],
        "affected_ranks": [],
        "snapshot_handle": {"epoch": seq, "tick": seq},
        "ledger_summary": {"injected_unfinished": []},
    }


def main() -> int:
    delivery_total = _positive_env(
        "SCHEDULER_MEMORY_DELIVERIES", 1_000_000, minimum=2)
    check_interval = _positive_env(
        "SCHEDULER_MEMORY_CHECK_INTERVAL", 10_000)
    request_count = delivery_total - 1
    scheduler = _MemoryScheduler(
        manifest={"requests": _LazyRequests(request_count)},
        config=_FixtureConfig(),
        mode="memory_fixture",
        profile_sink=_discard_row,
        online_stats_sink=_discard_row,
    )
    _assert_no_retired_containers(scheduler)

    for seq in range(delivery_total):
        scheduler.on_decision_batch(_delta(seq, request_count))
        scheduler.on_commit_ack({
            "schema_version": 1,
            "delivery_sequence": seq,
            "batch_id": seq,
            "success": True,
        })
        _assert_after_ack(scheduler, seq)
        if seq % check_interval == 0 or seq == delivery_total - 1:
            _assert_lifecycle(scheduler, seq, request_count)

    scheduler.verify_run_end()
    _assert_no_retired_containers(scheduler)
    rss_mib = _rss_mib()
    print(
        "long-run memory fixture PASS: {} deliveries, {} requests, rss={}"
        .format(delivery_total, request_count,
                "unavailable" if rss_mib is None else "{:.1f} MiB".format(rss_mib)),
        flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # noqa: BLE001 -- CLI fixture must fail closed
        print("long_run_memory_fixture: fatal: {}: {}".format(
            type(exc).__name__, exc), file=sys.stderr)
        sys.exit(1)
