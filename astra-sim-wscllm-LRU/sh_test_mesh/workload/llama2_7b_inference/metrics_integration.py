"""Metrics support utilities for the frozen schema (P1 KV delta journal).

Ported from astra-sim-face metrics_integration.py (implementation doc
sec.9.2).  Two pieces live here:

- :func:`load_chiplets_per_npu`: reads ``metrics_config.json`` for the
  chiplet-per-NPU default used by the online service metrics context.
- :class:`MemoryActionRecorder`: the streaming journal-first HBM delta
  ledger driven by the session KV manager (P1; see the class docstring
  for the journal semantics and the B2 remote-pool column).

Nothing here feeds back into request mapping, scheduling, KV management,
or dynamic graph construction.  The online GraphBatch routes materialize
their runtime inputs separately, and the run's ``metrics_manifest.json``
is synthesized by plan_materializer.py.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Optional

from metrics_schema import (  # noqa: E402
    MemoryDelta,
    MemoryMetricsObserver,
)


METRICS_CONFIG_PATH = Path(__file__).resolve().parent / "metrics_config.json"

__all__ = [
    "MemoryActionRecorder",
]


def load_chiplets_per_npu() -> int:
    try:
        with METRICS_CONFIG_PATH.open(encoding="utf-8") as source:
            config = json.load(source)
        return int(config.get("memory", {}).get("chiplets_per_npu", 4))
    except (OSError, ValueError, TypeError):
        return 4


class MemoryActionRecorder:
    """Planner memory delta sequence: streaming journal-first ledger.

    Two modes:

    - ``journal_path`` given (P1 authoritative HBM delta journal): every
      :meth:`record` appends exactly one JSON line to the journal file and
      flushes it (line-atomic crash consistency; a torn tail line is rejected
      with its line number on replay).  Nothing is retained in memory.
      :meth:`iter_journal_deltas` streams the journal back solely for the
      read-back test (sh_test_mesh/workload/llama2_7b_inference/
      test_kv_delta_journal.py:274); no manifest ``memory_actions`` rebuild
      chain exists in this repo.  The journal is the authoritative
      physical ledger replayed by the run-end checksum gate
      (``SessionKVCacheManager.verify_journal_checksum``).
    - ``journal_path`` omitted (legacy in-memory mode): deltas accumulate in
      ``self.deltas`` exactly as before and the manifest walks the list.

    The KV manager still calls :meth:`record` only *after* it applied the
    identical state change (doc sec.7.2); the recorder assigns the sequence
    index, forwards the delta to the read-only observer, and (in journal
    mode) streams the journal row.  Journal semantics (P1, four mandatory
    rules):

    - ``sequence`` is a dedicated monotonic counter (never ``len(list)``), so
      zero-delta records skipped by the manager leave the sequence contiguous
      under streaming.
    - ``transaction_id`` groups the rows of one manager-public mutation; 0
      marks pre-transaction rows (the construction-time weight preload).
      Transactions are driven by the manager (begin/finish/abort) and have no
      marker row — boundaries are recovered by grouping.
    - ``planner_time_ns`` must be non-decreasing across the journal; a
      decrease raises immediately (fail-closed: that is a timing bug, never
      silently clamped).
    - Each row carries the rank's ``before_bytes`` / ``after_bytes`` /
      ``capacity_bytes`` snapshots, so the file alone reconstructs the
      per-rank (weight, resident, reserved) ledger.
    - B2 (2026-09, two-state KV): rows additionally carry
      ``remote_delta_bytes`` and the per-rank ledger becomes
      ``(weight, resident, reserved, remote)``; the remote column accounts
      the bytes that rank stored into / restored from the shared remote
      memory pool (whole-session eviction, restore, terminal write-off).
      The frozen manifest ``MemoryDelta`` stream is untouched — remote is a
      journal-only column.
    """

    JOURNAL_SCHEMA_VERSION = 2
    JOURNAL_CHECKSUM_NAME = "kv_delta_journal_checksum.json"
    _JOURNAL_ROW_FIELDS = frozenset(
        {
            "schema_version",
            "sequence",
            "transaction_id",
            "planner_time_ns",
            "anchor_kind",
            "request_id",
            "session_id",
            "cause",
            "rank",
            "instance_index",
            "allocation_key",
            "weight_delta_bytes",
            "resident_kv_delta_bytes",
            "reserved_kv_delta_bytes",
            "remote_delta_bytes",
            "before_bytes",
            "after_bytes",
            "capacity_bytes",
        }
    )

    def __init__(
        self,
        observer: MemoryMetricsObserver,
        *,
        journal_path: Optional[Any] = None,
    ) -> None:
        self.observer = observer
        # 非流式兼容模式(journal_path=None)才驻留;journal 模式恒空。
        self.deltas: list[MemoryDelta] = []
        self.journal_path = Path(journal_path) if journal_path is not None else None
        self._journal_output: Optional[Any] = None
        # ② 独立单调计数器:流式下唯一可靠的 sequence 来源(改前
        # len(self.deltas) 在零 delta 记录被跳过/spool 化时不可靠)。
        self._sequence = 0
        self.record_count = 0
        # rank -> [weight, resident, reserved, remote]:journal 记账的逐 rank
        # 运行终态(commit 对账与行内 before/after 快照的唯一来源)。B2:
        # remote 列 = 该 rank 存入共享远端池的字节账面(journal-only)。
        self._rank_totals: dict[int, list[int]] = {}
        self._rank_capacity: dict[int, int] = {}
        self._rank_instance: dict[int, int] = {}
        # ① 时序护栏:journal 已写出的最大 planner_time_ns(仅 journal 模式
        # 维护;非单调即 raise,禁止钳位回退)。
        self._last_planner_time_ns: Optional[int] = None
        self._transaction_id_next = 1
        self._transaction_active: Optional[int] = None
        self._transaction_ranks: set[int] = set()
        self.transaction_count = 0
        if self.journal_path is not None:
            self.journal_path.parent.mkdir(parents=True, exist_ok=True)
            # 覆盖上一轮残留;常驻 fd + 每行 flush(§10.3 治理口径)。
            self._journal_output = self.journal_path.open("w", encoding="utf-8")

    @property
    def journal_enabled(self) -> bool:
        return self.journal_path is not None

    def initialize_rank(self, rank: int, capacity_bytes: int) -> None:
        self.observer.initialize_rank(rank, capacity_bytes)
        if self.journal_enabled:
            self._rank_totals[rank] = [0, 0, 0, 0]
            self._rank_capacity[rank] = int(capacity_bytes)

    def rank_totals(self, rank: int) -> tuple[int, int, int, int]:
        """Journal-side running (weight, resident, reserved, remote) for one
        rank (B2: remote pool column appended)."""

        totals = self._rank_totals.get(rank)
        if totals is None:
            raise RuntimeError(f"kv delta journal rank {rank} was never initialized")
        return (totals[0], totals[1], totals[2], totals[3])

    # ------------------------------------------------------------- ③ 事务 --

    def begin_transaction(self) -> int:
        """Open one manager-public mutation; returns its monotonic id."""

        if not self.journal_enabled:
            raise RuntimeError("kv delta journal transactions require journal mode")
        if self._transaction_active is not None:
            raise RuntimeError("nested kv delta journal transaction begin")
        transaction_id = self._transaction_id_next
        self._transaction_id_next += 1
        self._transaction_active = transaction_id
        self._transaction_ranks = set()
        self.transaction_count += 1
        return transaction_id

    def finish_transaction(self) -> tuple[int, frozenset[int]]:
        """Commit point: close the transaction, return (id, touched ranks)."""

        if self._transaction_active is None:
            raise RuntimeError("kv delta journal finish without an active transaction")
        transaction_id = self._transaction_active
        ranks = frozenset(self._transaction_ranks)
        self._transaction_active = None
        self._transaction_ranks = set()
        return transaction_id, ranks

    def abort_transaction(self) -> None:
        """Exception path: drop the registration (already-written rows stay;
        the enclosing run is failing closed anyway)."""

        self._transaction_active = None
        self._transaction_ranks = set()

    # ------------------------------------------------------------- 记录 --

    def record(
        self,
        *,
        planner_time_ns: int,
        anchor_kind: str,
        request_id: Optional[str],
        session_id: Optional[str],
        rank: int,
        allocation_key: str,
        weight_delta_bytes: int = 0,
        resident_kv_delta_bytes: int = 0,
        reserved_kv_delta_bytes: int = 0,
        remote_delta_bytes: int = 0,
        cause: str = "",
        instance_index: Optional[int] = None,
    ) -> None:
        planner_time_ns = int(planner_time_ns)
        if (
            self.journal_enabled
            and self._last_planner_time_ns is not None
            and planner_time_ns < self._last_planner_time_ns
        ):
            # ① 时序护栏:传入时刻早于 journal 已有最后时刻 = mutation
            # 时间戳 bug(如 release 复用了过期事件时间),fail-closed。
            raise RuntimeError(
                "kv delta journal planner_time_ns went backwards: "
                f"{planner_time_ns} after {self._last_planner_time_ns} "
                f"(allocation_key={allocation_key!r}, rank={rank}, "
                f"cause={cause!r}); this is a mutation-timestamp bug "
                "(fail-closed, never clamped)"
            )
        delta = MemoryDelta(
            sequence_index=self._sequence,
            planner_time_ns=planner_time_ns,
            anchor_kind=anchor_kind,
            trigger_queue_index=None,
            request_id=request_id,
            session_id=session_id,
            rank=rank,
            allocation_key=allocation_key,
            weight_delta_bytes=weight_delta_bytes,
            resident_kv_delta_bytes=resident_kv_delta_bytes,
            reserved_kv_delta_bytes=reserved_kv_delta_bytes,
            cause=cause,
        )
        # 观察者先行(与改前一致):观察者账本否决的行不进 journal。
        self.observer.record_delta(delta)
        sequence = self._sequence
        self._sequence += 1
        self.record_count += 1
        if not self.journal_enabled:
            self.deltas.append(delta)
            return
        before = self._rank_totals.get(rank)
        if before is None:
            raise RuntimeError(
                f"kv delta journal saw rank {rank} before initialize_rank"
            )
        component_deltas = (
            int(weight_delta_bytes),
            int(resident_kv_delta_bytes),
            int(reserved_kv_delta_bytes),
            int(remote_delta_bytes),
        )
        after = [
            before[0] + component_deltas[0],
            before[1] + component_deltas[1],
            before[2] + component_deltas[2],
            before[3] + component_deltas[3],
        ]
        if instance_index is None:
            instance_index = self._rank_instance.get(rank)
        else:
            self._rank_instance.setdefault(rank, int(instance_index))
        row = {
            "schema_version": self.JOURNAL_SCHEMA_VERSION,
            "sequence": sequence,
            "transaction_id": self._transaction_active or 0,
            "planner_time_ns": planner_time_ns,
            "anchor_kind": anchor_kind,
            "request_id": request_id,
            "session_id": session_id,
            "cause": cause,
            "rank": int(rank),
            "instance_index": instance_index,
            "allocation_key": allocation_key,
            "weight_delta_bytes": component_deltas[0],
            "resident_kv_delta_bytes": component_deltas[1],
            "reserved_kv_delta_bytes": component_deltas[2],
            "remote_delta_bytes": component_deltas[3],
            "before_bytes": {
                "weight": before[0],
                "resident": before[1],
                "reserved": before[2],
                "remote": before[3],
            },
            "after_bytes": {
                "weight": after[0],
                "resident": after[1],
                "reserved": after[2],
                "remote": after[3],
            },
            "capacity_bytes": self._rank_capacity[rank],
        }
        # 行级原子:一行完整 JSON + 立即 flush;断行由重放方 fail-closed
        # 报行号(本进程崩溃时尾部至多损失正在写的这一行)。
        self._journal_output.write(json.dumps(row, sort_keys=True) + "\n")
        self._journal_output.flush()
        self._rank_totals[rank] = after
        self._last_planner_time_ns = planner_time_ns
        if self._transaction_active is not None:
            self._transaction_ranks.add(int(rank))

    # ----------------------------------------------------- 流式回读/重放 --

    def _read_journal_row(self, line_number: int, line: Any) -> dict[str, Any]:
        """Decode and shape-check one journal line (fail-closed, with line no)."""

        try:
            row = json.loads(line)
        except (UnicodeDecodeError, ValueError) as error:
            raise RuntimeError(
                f"kv delta journal line {line_number} is corrupt "
                f"(fail-closed): {error}"
            ) from None
        if not isinstance(row, dict):
            raise RuntimeError(
                f"kv delta journal line {line_number} is not a JSON object"
            )
        missing = self._JOURNAL_ROW_FIELDS - set(row)
        unknown = set(row) - self._JOURNAL_ROW_FIELDS
        if missing or unknown:
            raise RuntimeError(
                f"kv delta journal line {line_number} has wrong fields: "
                f"missing={sorted(missing)}, unknown={sorted(unknown)}"
            )
        if row["schema_version"] != self.JOURNAL_SCHEMA_VERSION:
            raise RuntimeError(
                f"kv delta journal line {line_number}: unsupported "
                f"schema_version {row['schema_version']!r}"
            )
        return row

    def iter_journal_deltas(self):
        """Stream the journal back as ``MemoryDelta`` rows.

        Read-back use only: the sole caller is the journal round-trip test
        (sh_test_mesh/workload/llama2_7b_inference/test_kv_delta_journal.py:274).
        No manifest ``memory_actions`` rebuild chain exists in this repo
        (the former "(manifest source)" claim was stale; corrected
        2026-09-25).
        """

        if not self.journal_enabled:
            raise RuntimeError("iter_journal_deltas requires journal mode")
        with self.journal_path.open("r", encoding="utf-8") as source:
            for line_number, line in enumerate(source, 1):
                if not line.strip():
                    raise RuntimeError(
                        f"kv delta journal line {line_number} is blank "
                        "(fail-closed)"
                    )
                row = self._read_journal_row(line_number, line)
                yield MemoryDelta(
                    sequence_index=row["sequence"],
                    planner_time_ns=row["planner_time_ns"],
                    anchor_kind=row["anchor_kind"],
                    trigger_queue_index=None,
                    request_id=row["request_id"],
                    session_id=row["session_id"],
                    rank=row["rank"],
                    allocation_key=row["allocation_key"],
                    weight_delta_bytes=row["weight_delta_bytes"],
                    resident_kv_delta_bytes=row["resident_kv_delta_bytes"],
                    reserved_kv_delta_bytes=row["reserved_kv_delta_bytes"],
                    cause=row["cause"],
                )

    def replay_journal(self) -> dict[str, Any]:
        """Stream-replay the whole journal, validating every invariant.

        Checks per line: schema/field shape, ``sequence`` contiguity from 0,
        per-rank ``before_bytes`` == running totals, ``after_bytes`` ==
        before + deltas, ``planner_time_ns`` non-decreasing, and
        ``transaction_id`` non-decreasing with each id's rows contiguous.
        Any violation raises with the offending line number; a torn/partial
        tail line is treated the same way (crash tolerance = fail-closed
        reporting, never silent truncation).
        """

        if not self.journal_enabled:
            raise RuntimeError("replay_journal requires journal mode")
        digest = hashlib.sha256()
        totals: dict[int, list[int]] = {}
        capacity: dict[int, int] = {}
        line_count = 0
        expected_sequence = 0
        last_time_ns: Optional[int] = None
        current_transaction: Optional[int] = None
        transaction_ids: set[int] = set()
        with self.journal_path.open("rb") as source:
            for line_number, raw in enumerate(source, 1):
                digest.update(raw)
                row = self._read_journal_row(
                    line_number, raw.decode("utf-8", errors="strict"))
                line_count += 1
                if row["sequence"] != expected_sequence:
                    raise RuntimeError(
                        f"kv delta journal line {line_number}: sequence "
                        f"{row['sequence']} breaks contiguity (expected "
                        f"{expected_sequence})"
                    )
                expected_sequence += 1
                rank = row["rank"]
                running = totals.get(rank)
                before = row["before_bytes"]
                if running is None:
                    running = [0, 0, 0, 0]
                    totals[rank] = running
                    capacity[rank] = row["capacity_bytes"]
                elif capacity[rank] != row["capacity_bytes"]:
                    raise RuntimeError(
                        f"kv delta journal line {line_number}: capacity_bytes "
                        f"changed for rank {rank}: {capacity[rank]} -> "
                        f"{row['capacity_bytes']}"
                    )
                if (
                    before["weight"],
                    before["resident"],
                    before["reserved"],
                    before["remote"],
                ) != tuple(running):
                    raise RuntimeError(
                        f"kv delta journal line {line_number}: rank {rank} "
                        f"before_bytes {before} does not match the replayed "
                        f"running totals {tuple(running)}"
                    )
                component_after = [
                    running[0] + row["weight_delta_bytes"],
                    running[1] + row["resident_kv_delta_bytes"],
                    running[2] + row["reserved_kv_delta_bytes"],
                    running[3] + row["remote_delta_bytes"],
                ]
                after = row["after_bytes"]
                if (
                    after["weight"],
                    after["resident"],
                    after["reserved"],
                    after["remote"],
                ) != tuple(component_after):
                    raise RuntimeError(
                        f"kv delta journal line {line_number}: rank {rank} "
                        f"after_bytes {after} != before+deltas "
                        f"{tuple(component_after)}"
                    )
                if last_time_ns is not None and row["planner_time_ns"] < last_time_ns:
                    raise RuntimeError(
                        f"kv delta journal line {line_number}: "
                        f"planner_time_ns {row['planner_time_ns']} < previous "
                        f"{last_time_ns} (non-monotonic)"
                    )
                last_time_ns = row["planner_time_ns"]
                transaction_id = row["transaction_id"]
                if transaction_id != current_transaction:
                    if transaction_id in transaction_ids:
                        raise RuntimeError(
                            f"kv delta journal line {line_number}: rows of "
                            f"transaction {transaction_id} are not contiguous"
                        )
                    if (
                        current_transaction is not None
                        and transaction_id < current_transaction
                    ):
                        raise RuntimeError(
                            f"kv delta journal line {line_number}: "
                            f"transaction_id {transaction_id} < previous "
                            f"{current_transaction} (non-monotonic)"
                        )
                    transaction_ids.add(transaction_id)
                    current_transaction = transaction_id
                totals[rank] = component_after
        ranks = {
            rank: {
                "weight": state[0],
                "resident": state[1],
                "reserved": state[2],
                "remote": state[3],
                "physical": state[0] + state[1],
                "capacity_bytes": capacity[rank],
            }
            for rank, state in sorted(totals.items())
        }
        return {
            "line_count": line_count,
            "sha256": digest.hexdigest(),
            "transaction_count": len(transaction_ids),
            "max_transaction_id": max(transaction_ids) if transaction_ids else 0,
            "ranks": ranks,
        }

    def close_journal(self) -> None:
        if self._journal_output is not None:
            self._journal_output.close()
            self._journal_output = None

