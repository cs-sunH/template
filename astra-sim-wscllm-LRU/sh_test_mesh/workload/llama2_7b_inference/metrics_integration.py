"""Legacy static-ET metrics compatibility utilities for the frozen schema.

Ported from astra-sim-face metrics_integration.py (implementation doc sec.9.2).
These observational helpers can build a metrics sidecar and digest records for
explicitly supplied historic ET files.  They are not imported by the online
GraphBatch routes, whose runtime inputs are materialized separately.  Nothing
here feeds back into request mapping, scheduling, KV management, or dynamic
graph construction.

WSC-LLM differences versus the FACE original:

- ``REPO_VARIANT`` is ``astra-sim-wscllm``.
The manifest is exactly the frozen-schema product of
:class:`metrics_schema.MetricManifestBuilder`; no extra keys are injected
beyond what the frozen ``MetricsManifest`` dataclass emits.

Legacy digest recipes:

- ``trace_digest``: per rank ``<rank>:<sha256 hexdigest of the .et bytes>``
  lines joined with ``\\n``, then SHA256 of that UTF-8 text.  The run script
  recomputes exactly this recipe before launching the simulator (doc sec.10).
- ``request_mapping_digest``: SHA256 of the canonical JSON
  (``sort_keys``, compact separators) of the original manifest's ``requests``
  array, which carries the WSC-LLM prefill/decode mapping and KV placement.
- ``kv_event_digest``: SHA256 of the canonical JSON of the KV event payload
  (session-LRU: every ``KVCacheEvent`` field row in order).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from metrics_schema import (  # noqa: E402
    Arrival,
    EVENT_DECODE_END,
    EVENT_DECODE_START,
    EVENT_MEMORY_ANCHOR_COMPLETE,
    EVENT_PREFILL_END,
    EVENT_PREFILL_START,
    MemoryDelta,
    MemoryMetricsObserver,
    MemoryProjection,
    MetricManifestBuilder,
    RequestMetadata,
    RUN_MODE_SERVICE,
)


REPO_VARIANT = "astra-sim-wscllm"
METRICS_CONFIG_PATH = Path(__file__).resolve().parent / "metrics_config.json"

# Anchor kinds follow the doc sec.7.8 table; anything else is a nearest-stage
# fallback and is tagged anchor_quality=stage_boundary in the manifest.
EXACT_ANCHOR_KINDS = frozenset(
    {"tick_zero", "prefill_start", "transfer_complete", "completion"}
)

__all__ = [
    "EVENT_DECODE_END",
    "EVENT_DECODE_START",
    "EVENT_MEMORY_ANCHOR_COMPLETE",
    "EVENT_PREFILL_END",
    "EVENT_PREFILL_START",
    "MemoryActionRecorder",
    "PlannerLutStatsAccumulator",
    "ServiceMetrics",
    "canonical_json",
    "compute_trace_digest",
    "resolve_metrics_detail",
    "write_planner_lut_stats",
]


def kv_bin_power_of_two(value: int) -> int:
    """Bin a KV length into its power-of-two ceiling (0 stays bin 0)."""

    value = int(value)
    if value <= 0:
        return 0
    return 1 << (value - 1).bit_length()


class PlannerLutStatsAccumulator:
    """Streaming planner-iteration aggregates (doc sec.8.8).

    The service planner notifies one LUT lookup per planning iteration through
    :meth:`record_lut_iteration`; only per-cell count/sum/min/max are kept, so
    memory stays O(cells) regardless of iteration count.  Cells are keyed by
    ``(phase, tp_degree, batch, kv_bin)`` where phase is ``prefill`` /
    ``decode`` / ``mixed`` (a mixed iteration carries both a prefill chunk and
    a decode batch in the FACE LUT model), batch is the prefill chunk for
    prefill cells and the decode batch otherwise, and kv_bin is the
    power-of-two ceiling of the decode KV length.  These records are a
    planner-LUT proxy (``source=planner_lut``), never the paper's primary
    iteration-time source.
    """

    def __init__(self) -> None:
        # (phase, tp_degree, batch, kv_bin, prefill_chunk) -> [count,sum,min,max]
        self._cells: dict[tuple[str, int, int, int, int], list[int]] = {}

    def record_lut_iteration(
        self, lut_entry: Any, start_ns: int, end_ns: int
    ) -> None:
        p_chunk = int(lut_entry.p_chunk)
        d_batch = int(lut_entry.d_batch)
        if p_chunk > 0 and d_batch > 0:
            phase = "mixed"
        elif p_chunk > 0:
            phase = "prefill"
        else:
            phase = "decode"
        batch = p_chunk if phase == "prefill" else d_batch
        key = (
            phase,
            int(lut_entry.instance_size),
            batch,
            kv_bin_power_of_two(int(lut_entry.d_token)),
            p_chunk,
        )
        iteration_time_ns = int(end_ns) - int(start_ns)
        cell = self._cells.get(key)
        if cell is None:
            self._cells[key] = [1, iteration_time_ns, iteration_time_ns, iteration_time_ns]
        else:
            cell[0] += 1
            cell[1] += iteration_time_ns
            cell[2] = min(cell[2], iteration_time_ns)
            cell[3] = max(cell[3], iteration_time_ns)

    def to_records(self, *, repo_variant: str = REPO_VARIANT) -> list[dict[str, Any]]:
        records = []
        for (phase, tp_degree, batch, kv_bin, p_chunk), cell in sorted(
            self._cells.items()
        ):
            records.append(
                {
                    "schema": 1,
                    "type": "planner_lut_iteration_stats",
                    "source": "planner_lut",
                    "repo_variant": repo_variant,
                    "phase": phase,
                    "tp_degree": tp_degree,
                    "batch": batch,
                    "kv_bin": kv_bin,
                    "prefill_chunk_tokens": p_chunk,
                    "count": cell[0],
                    "sum_iteration_time_ns": cell[1],
                    "min_iteration_time_ns": cell[2],
                    "max_iteration_time_ns": cell[3],
                }
            )
        return records


def write_planner_lut_stats(
    accumulator: PlannerLutStatsAccumulator,
    *,
    output_dir: Path,
    repo_variant: str = REPO_VARIANT,
) -> Path:
    """Write the planner_lut_stats.json sidecar and echo every record as a
    single-line ``[METRIC]`` JSON record (doc sec.8.8/11.1)."""

    records = accumulator.to_records(repo_variant=repo_variant)
    sidecar = output_dir / "planner_lut_stats.json"
    sidecar.write_text(
        json.dumps(
            {
                "schema": 1,
                "source": "planner_lut",
                "repo_variant": repo_variant,
                "records": records,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    for record in records:
        print("[METRIC] " + json.dumps(record, separators=(",", ":")))
    return sidecar


def canonical_json(payload: Any) -> str:
    """Deterministic JSON serialization used by every metrics digest."""

    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def compute_trace_digest(et_paths_by_rank: Mapping[int, Path]) -> str:
    """Hash explicitly supplied legacy ET paths in deterministic rank order."""

    per_rank = []
    for rank, path in sorted(et_paths_by_rank.items()):
        per_rank.append(f"{rank}:{hashlib.sha256(path.read_bytes()).hexdigest()}")
    return _sha256_hex("\n".join(per_rank))


def request_mapping_digest(request_records: Sequence[Mapping[str, Any]]) -> str:
    return _sha256_hex(canonical_json(list(request_records)))


def kv_event_digest(kv_payload: Any) -> str:
    return _sha256_hex(canonical_json(kv_payload))


def kv_event_payload_session_lru(events: Sequence[Any]) -> list[list[Any]]:
    """One row per KVCacheEvent, in event order, mirroring kv_cache_events.csv."""

    return [
        [
            event.event_index,
            event.planner_time_ns,
            event.phase,
            event.event_type,
            event.reason,
            event.trigger_request_id,
            event.session_id,
            event.source_instance_index,
            event.target_instance_index,
            event.context_tokens,
            event.total_bytes,
            list(event.shard_bytes),
            event.last_completion_ns,
            list(event.instance_remaining_before_bytes),
            list(event.instance_remaining_after_bytes),
            list(event.insufficient_ranks),
        ]
        for event in events
    ]


def load_chiplets_per_npu() -> int:
    try:
        with METRICS_CONFIG_PATH.open(encoding="utf-8") as source:
            config = json.load(source)
        return int(config.get("memory", {}).get("chiplets_per_npu", 4))
    except (OSError, ValueError, TypeError):
        return 4


def resolve_metrics_detail(
    cli_value: Optional[str], environ: Mapping[str, str]
) -> str:
    """Resolve the generator metrics switch: CLI > METRICS_DETAIL >
    ENABLE_METRICS > default on/full (aligned with the online runner metrics switch)."""

    detail = cli_value if cli_value is not None else environ.get("METRICS_DETAIL")
    if detail is None:
        enable = environ.get("ENABLE_METRICS", "1").strip().lower()
        detail = "off" if enable in {"0", "off", "false", "no"} else "full"
    detail = detail.strip().lower()
    if detail not in {"off", "summary", "full"}:
        raise SystemExit(
            f"invalid metrics detail {detail!r}; expected off|summary|full"
        )
    return detail


class MemoryActionRecorder:
    """Planner memory delta sequence: streaming journal-first ledger.

    Two modes:

    - ``journal_path`` given (P1 authoritative HBM delta journal): every
      :meth:`record` appends exactly one JSON line to the journal file and
      flushes it (line-atomic crash consistency; a torn tail line is rejected
      with its line number on replay).  Nothing is retained in memory — the
      manifest ``memory_actions`` stream is rebuilt by streaming the file back
      (:meth:`iter_journal_deltas`).  The journal is the authoritative
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
    - B2 (2026-09, three-state KV): rows additionally carry
      ``remote_delta_bytes`` and the per-rank ledger becomes
      ``(weight, resident, reserved, remote)``; the remote column accounts
      the bytes that rank stored into / restored from the shared remote
      memory pool (suffix/full eviction, restore, terminal write-off).  The
      frozen manifest ``MemoryDelta`` stream is untouched — remote is a
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
        """Stream the journal back as ``MemoryDelta`` rows (manifest source)."""

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


def _build_request_metadata(config: Any, plan: Any) -> list[RequestMetadata]:
    """Manifest request records (doc sec.4.2/4.4) for the whole queue.

    Turn-0 arrivals are absolute session arrivals; later turns reference the
    same session's previous turn request by ``queue_index`` so the simulator
    resolves them from the parent's *actual* completion (doc sec.3.1).
    """

    queue_by_session_turn = {
        (spec.session_id, spec.turn_index): index
        for index, spec in enumerate(config.request_queue)
    }
    group_by_index = dict(enumerate(config.inference_groups))
    plans_by_queue = {request_plan.queue_index: request_plan for request_plan in plan.requests}
    metadata: list[RequestMetadata] = []
    for index, spec in enumerate(config.request_queue):
        request_plan = plans_by_queue.get(index)
        if request_plan is None:
            raise RuntimeError(
                f"request queue_index {index} has no plan; the metrics manifest "
                "requires every queued request to be planned"
            )
        if spec.turn_index == 0:
            if spec.session_arrival_time_ns is None:
                raise RuntimeError("first request lost its session arrival time")
            arrival = Arrival.absolute(spec.session_arrival_time_ns)
        else:
            parent_index = queue_by_session_turn.get(
                (spec.session_id, spec.turn_index - 1)
            )
            if parent_index is None or spec.inter_request_interval_ns is None:
                raise RuntimeError(
                    f"request {spec.request_id} lost its previous-turn parent"
                )
            arrival = Arrival.after_request(
                parent_index, spec.inter_request_interval_ns
            )
        metadata.append(
            RequestMetadata(
                queue_index=index,
                request_id=request_plan.request_id,
                session_id=request_plan.session_id,
                turn_index=request_plan.turn_index,
                arrival=arrival,
                prefill_instance=request_plan.prefill_instance_index,
                prefill_ranks=tuple(
                    group_by_index[request_plan.prefill_instance_index].ranks
                ),
                decode_instance=request_plan.decode_instance_index,
                decode_ranks=tuple(
                    group_by_index[request_plan.decode_instance_index].ranks
                ),
            )
        )
    return metadata


class ServiceMetrics:
    """Per-generation metrics context for one WSC-LLM service trace run."""

    def __init__(
        self,
        detail: str,
        *,
        chiplets_per_npu: Optional[int] = None,
        journal_path: Optional[Any] = None,
    ) -> None:
        if detail not in {"summary", "full"}:
            raise ValueError("ServiceMetrics requires detail summary|full")
        self.detail = detail
        self.chiplets_per_npu = (
            load_chiplets_per_npu() if chiplets_per_npu is None else chiplets_per_npu
        )
        self.memory = MemoryActionRecorder(
            MemoryMetricsObserver(self.chiplets_per_npu),
            journal_path=journal_path,
        )
        # (rank, node_id, event_code, subject_id) in emission order.
        self.events: list[tuple[int, int, int, int]] = []

    def add_event(
        self, rank: int, node_id: int, event_code: int, subject_id: int
    ) -> None:
        self.events.append((rank, node_id, event_code, subject_id))

    def write_manifest(
        self,
        *,
        output_dir: Path,
        config: Any,
        plan: Any,
        node_count_by_rank: Mapping[int, int],
        et_paths_by_rank: Mapping[int, Path],
        request_records: Sequence[Mapping[str, Any]],
        kv_digest_payload: Any,
    ) -> Path:
        """Assemble, validate (doc sec.4.5), and write metrics_manifest.json."""

        if self.memory.journal_enabled:
            # P1 流式:memory_actions 改为从 journal 文件回读(逐行迭代,
            # 不驻留内存),冻结 schema 的 action 字段一字不改。
            if self.memory.record_count == 0:
                raise RuntimeError(
                    "no memory deltas were recorded; the session KV manager must "
                    "run under set_metrics_observer() before writing the metrics "
                    "manifest"
                )
            deltas = self.memory.iter_journal_deltas()
        else:
            if not self.memory.deltas:
                raise RuntimeError(
                    "no memory deltas were recorded; the session KV manager must "
                    "run under set_metrics_observer() before writing the metrics "
                    "manifest"
                )
            deltas = iter(self.memory.deltas)
        request_id_to_queue = {
            spec.request_id: index for index, spec in enumerate(config.request_queue)
        }
        builder = MetricManifestBuilder(
            repo_variant=REPO_VARIANT,
            run_mode=RUN_MODE_SERVICE,
            npus_count=config.npus_count,
            mesh_rows=config.hardware.mesh_rows,
            mesh_columns=config.hardware.mesh_cols,
            node_count_by_rank=dict(node_count_by_rank),
            memory_projection=MemoryProjection(
                chiplets_per_npu=self.chiplets_per_npu
            ),
        )
        for metadata in _build_request_metadata(config, plan):
            builder.add_request(metadata)
        for rank, node_id, event_code, subject_id in self.events:
            builder.add_node_event(rank, node_id, event_code, subject_id)
        builder.set_digests(
            trace_digest=compute_trace_digest(et_paths_by_rank),
            request_mapping_digest=request_mapping_digest(request_records),
            kv_event_digest=kv_event_digest(kv_digest_payload),
        )
        for delta in deltas:
            action = delta.to_dict()
            if delta.request_id is not None:
                action["trigger_queue_index"] = request_id_to_queue.get(
                    delta.request_id
                )
            action["anchor_quality"] = (
                "exact"
                if delta.anchor_kind in EXACT_ANCHOR_KINDS
                else "stage_boundary"
            )
            builder.add_memory_action(action)
        memory_result = self.memory.observer.finalize()
        for rank, rank_result in sorted(memory_result.ranks.items()):
            builder.add_planner_memory_peak(rank_result.to_dict())
        manifest = builder.build()
        manifest_path = output_dir / "metrics_manifest.json"
        # Compact serialization: the memory action stream is machine-read
        # (C++ MetricCollector / post-processing), and pretty-printing it
        # would grow the sidecar by several times on large workloads.
        manifest_path.write_text(
            json.dumps(manifest.to_dict(), separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        return manifest_path
