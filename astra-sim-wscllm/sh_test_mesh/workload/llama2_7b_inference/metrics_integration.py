"""Service-route metrics integration helpers.

``MemoryActionRecorder`` is the P1 authoritative HBM delta journal shared by
``session_kv_manager`` and the online routes (doc sec.7.2/10.3); the journal
file is the authoritative physical ledger, streamed back line by line at
replay time (``replay_journal``).  ``load_chiplets_per_npu`` resolves the
per-NPU chiplet count from ``metrics_config.json``.  Nothing here feeds back
into request mapping, scheduling, KV management, or dynamic graph
construction.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Optional

from metrics_schema import (  # noqa: E402
    MemoryDelta,
    MemoryMetricsObserver,
)


METRICS_CONFIG_PATH = Path(__file__).resolve().parent / "metrics_config.json"

# Single source of truth for the config schema version: the same constant
# the metrics contract tests pin (metrics_schema.SCHEMA_VERSION), so a
# future version bump updates both consumers together (R5-F2, 2026-10-01).
from metrics_schema import SCHEMA_VERSION as _EXPECTED_SCHEMA_VERSION

__all__ = [
    "MemoryActionRecorder",
    "load_chiplets_per_npu",
]


def load_chiplets_per_npu() -> int:
    # 配置损坏/缺失不是静默场景：stderr 记录原因后回退默认值 4（与
    # metrics_config.json 当前值相同，但掩盖配置错误会让排查失去线索）。
    try:
        with METRICS_CONFIG_PATH.open(encoding="utf-8") as source:
            config = json.load(source)
        # Schema gate (R4-blind-F2, 2026-10-01): schema_version was
        # previously consumed by nothing at runtime, so a future format
        # revision would silently read the new file under the old key
        # shape. A missing or non-1 version walks this function's existing
        # stderr + default-4 channel like every other malformed-config
        # path.
        schema_version = config.get("schema_version")
        if schema_version != _EXPECTED_SCHEMA_VERSION:
            print(
                f"metrics_integration: {METRICS_CONFIG_PATH} has "
                f"schema_version={schema_version!r} (expected "
                f"{_EXPECTED_SCHEMA_VERSION}); falling back to "
                "chiplets_per_npu=4",
                file=sys.stderr,
            )
            return 4
        memory = config.get("memory")
        if not isinstance(memory, dict):
            print(
                f"metrics_integration: {METRICS_CONFIG_PATH} lacks a valid "
                "'memory' object; falling back to chiplets_per_npu=4",
                file=sys.stderr,
            )
            return 4
        chiplets = memory.get("chiplets_per_npu", 4)
        # Report 4.3 fix (2026-09-29): the bare int() silently coerced bools
        # (True -> 1), floats (4.5 -> 4) and numeric strings ("4" -> 4), so
        # a corrupted config value passed as a valid chiplet count.  The
        # count must be a real integer (bool is an int subclass and is
        # rejected explicitly); a wrong-shaped value walks this function's
        # existing stderr + default-4 channel like every other
        # malformed-config path above.
        if (
            isinstance(chiplets, bool)
            or not isinstance(chiplets, int)
            or chiplets < 1
        ):
            # Value-domain twin of the type gate (2026-09-30): 0/negative is
            # the same corrupted-config shape and previously slipped through
            # to die inside MemoryMetricsObserver's own _require_int, far
            # from the config file; route it through this function's
            # stderr + default-4 channel like every other malformed value.
            print(
                f"metrics_integration: {METRICS_CONFIG_PATH} has a "
                f"non-positive 'chiplets_per_npu' ({chiplets!r}); falling "
                "back to chiplets_per_npu=4",
                file=sys.stderr,
            )
            return 4
        return chiplets
    except (OSError, ValueError, TypeError, AttributeError) as error:
        print(
            f"metrics_integration: cannot read chiplets_per_npu from "
            f"{METRICS_CONFIG_PATH} ({error}); falling back to 4",
            file=sys.stderr,
        )
        return 4


class MemoryActionRecorder:
    """Planner memory delta sequence: streaming journal ledger.

    Every :meth:`record` appends exactly one JSON line to the journal file
    and flushes it (line-atomic crash consistency; a torn tail line is
    rejected with its line number on replay).  Nothing is retained in
    memory -- the journal is the authoritative physical ledger replayed by
    the run-end checksum gate
    (``SessionKVCacheManager.verify_journal_checksum``).

    The KV manager still calls :meth:`record` only *after* it applied the
    identical state change (doc sec.7.2); the recorder assigns the sequence
    index, forwards the delta to the read-only observer, and streams the
    journal row.  Journal semantics (P1, four mandatory rules):

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
    """

    JOURNAL_SCHEMA_VERSION = 1
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
        self.journal_path = Path(journal_path) if journal_path is not None else None
        self._journal_output: Optional[Any] = None
        # ② 独立单调计数器:流式下唯一可靠的 sequence 来源。
        self._sequence = 0
        self.record_count = 0
        # rank -> [weight, resident, reserved]:journal 记账的逐 rank 运行
        # 终态(commit 对账与行内 before/after 快照的唯一来源)。
        self._rank_totals: dict[int, list[int]] = {}
        self._rank_capacity: dict[int, int] = {}
        self._rank_instance: dict[int, int] = {}
        # ① 时序护栏:journal 已写出的最大 planner_time_ns(仅 journal 模式
        # 维护;非单调即 raise,禁止钳位回退)。
        self._last_planner_time_ns: Optional[int] = None
        self._transaction_id_next = 1
        self._transaction_active: Optional[int] = None
        self._transaction_ranks: set[int] = set()
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
            self._rank_totals[rank] = [0, 0, 0]
            self._rank_capacity[rank] = int(capacity_bytes)

    def rank_totals(self, rank: int) -> tuple[int, int, int]:
        """Journal-side running (weight, resident, reserved) for one rank."""

        totals = self._rank_totals.get(rank)
        if totals is None:
            raise RuntimeError(f"kv delta journal rank {rank} was never initialized")
        return (totals[0], totals[1], totals[2])

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
            # legacy 内存驻留模式已删除:journal 路径是唯一权威账本,
            # 无 journal 的 record 一律 fail-closed。
            raise RuntimeError(
                "kv delta journal mode is required (legacy in-memory "
                "deltas mode removed; construct with journal_path)"
            )
        before = self._rank_totals.get(rank)
        if before is None:
            raise RuntimeError(
                f"kv delta journal saw rank {rank} before initialize_rank"
            )
        component_deltas = (
            int(weight_delta_bytes),
            int(resident_kv_delta_bytes),
            int(reserved_kv_delta_bytes),
        )
        after = [
            before[0] + component_deltas[0],
            before[1] + component_deltas[1],
            before[2] + component_deltas[2],
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
            "before_bytes": {
                "weight": before[0],
                "resident": before[1],
                "reserved": before[2],
            },
            "after_bytes": {
                "weight": after[0],
                "resident": after[1],
                "reserved": after[2],
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

    def _read_journal_row(self, line_number: int, raw: Any) -> dict[str, Any]:
        """Decode and shape-check one journal line (fail-closed, with line no).

        D2 fix (2026-09-28): the UTF-8 decode moved inside this method's
        try (a bad-byte line used to escape the replay loop as a bare
        UnicodeDecodeError, outside this line-numbered fail-closed
        channel), and the row shape check now covers the sub-objects and
        numeric fields the replay arithmetic touches --
        ``before_bytes``/``after_bytes`` must be objects carrying exactly
        the three non-negative integer components, and
        sequence/transaction_id/planner_time_ns/rank/capacity_bytes/
        deltas must be integers (bool excluded -- F2_PY 2026-09-29 closed
        the capacity_bytes/schema_version residues of the same list).  The
        replay loop previously escaped with bare
        KeyError/TypeError on such structurally corrupt rows.
        """

        try:
            line = raw.decode("utf-8", errors="strict")
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
        # F2_PY fix (2026-09-29, RV5-F5 / SC-1 same family): the former
        # `!= 1` comparison swallowed bool masquerades (True == 1); the
        # schema version must be the literal integer.
        schema_version = row["schema_version"]
        if (isinstance(schema_version, bool)
                or not isinstance(schema_version, int)
                or schema_version != self.JOURNAL_SCHEMA_VERSION):
            raise RuntimeError(
                f"kv delta journal line {line_number}: unsupported "
                f"schema_version {schema_version!r}"
            )
        # F2_PY fix (2026-09-29, RV5-F5): capacity_bytes joined the int
        # gate list -- a bool/non-int value used to flow into the replay's
        # per-rank capacity audit summary (and the later drift gate
        # `!=` let bool/int rows inter-penetrate).
        for name in ("sequence", "transaction_id", "planner_time_ns",
                     "rank", "capacity_bytes", "weight_delta_bytes",
                     "resident_kv_delta_bytes", "reserved_kv_delta_bytes"):
            value = row[name]
            if isinstance(value, bool) or not isinstance(value, int):
                raise RuntimeError(
                    f"kv delta journal line {line_number}: {name} must be "
                    f"an integer (got {value!r})"
                )
        for name in ("before_bytes", "after_bytes"):
            sub = row[name]
            if (not isinstance(sub, dict)
                    or set(sub) != {"weight", "resident", "reserved"}):
                raise RuntimeError(
                    f"kv delta journal line {line_number}: {name} must be "
                    "an object with exactly weight/resident/reserved"
                )
            for key in ("weight", "resident", "reserved"):
                value = sub[key]
                if (isinstance(value, bool) or not isinstance(value, int)
                        or value < 0):
                    raise RuntimeError(
                        f"kv delta journal line {line_number}: {name}."
                        f"{key} must be a non-negative integer "
                        f"(got {value!r})"
                    )
        return row

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
                row = self._read_journal_row(line_number, raw)
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
                    running = [0, 0, 0]
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
                ]
                after = row["after_bytes"]
                if (
                    after["weight"],
                    after["resident"],
                    after["reserved"],
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
                "physical": state[0] + state[1],
                "capacity_bytes": capacity[rank],
            }
            for rank, state in sorted(totals.items())
        }
        return {
            "line_count": line_count,
            "sha256": digest.hexdigest(),
            # D2 fix (2026-09-28): transaction_count 不再计入 transaction_id=0
            # 哨兵（0 = 事务前的构造期权重预载行，不是 manager 公开事务）
            # ——旧行为让每本 journal 的该审计字段恒 +1。contiguity/单调
            # 校验仍覆盖全部 id（含 0）；max_transaction_id 不受影响
            # （真实 id 从 1 起，0 永不可能是最大值）。
            "transaction_count": sum(
                1 for tid in transaction_ids if tid != 0),
            "max_transaction_id": max(transaction_ids) if transaction_ids else 0,
            "ranks": ranks,
        }

    def close_journal(self) -> None:
        if self._journal_output is not None:
            self._journal_output.close()
            self._journal_output = None

