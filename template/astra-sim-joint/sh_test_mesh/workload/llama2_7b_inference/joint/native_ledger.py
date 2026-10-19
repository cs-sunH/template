"""Typed ctypes facade for the native joint dual-ledger receiver C ABI.

joint 加速改造 v4（2026-10-12）N 主线：协议 v4 双账本（WorkLedger +
PipelineLedgerV4）的接收/校验/应用/提交全部驻留原生库
（extern/joint_ledger，libjoint_ledger_cabi.so）。本模块是 Python 侧唯
一桥接层：

* ``scan_request``：桥层用——结构扫描定位 request 顶层 ``work_ledger``
  成员，返回 outer 拼接字节 + 子段原始字节（零解码、零状态）；
* ``NativeDualLedgerReceiver``：调度器用——prepare（SAX typed 解析 +
  完整校验 + 双账本 COW 私有事务）/commit（原子发布）与查询面
  （pricing 前沿、计数器、守恒报告、离线导出）。每次交付常数次 FFI，
  不逐事件跨语言、不在 Python 重建 work_ledger DTO。

历史形态兼容入口 ``receive_payload_dict``：嵌入调用方/离线 harness 直
接持 work_ledger dict 时，紧凑序列化后进同一原生接收器（同一语义核
心，不是第二套状态机）。Python 参考解码器（work_ledger.py /
pipeline_ledger.py 的离线 decoder 面）保留用于等价性对照测试。

幂等重放门：原生侧缓存最后接受交付的原始子段字节做字节比对；生产者
（nlohmann dump）对相同内容确定性输出相同字节，重放即零变更。
"""

from __future__ import annotations

import ctypes
import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterator, Mapping, Optional, Tuple

from joint.work_ledger import (
    CommittedWork,
    LedgerPricingSnapshot,
    LedgerProjectionRow,
    WorkAmount,
    WorkIdentity,
)


class NativeLedgerError(RuntimeError):
    """C ABI load, scan, parse, validation, or state failure (fail-closed)."""


# ---------------------------------------------------------------- loader --

def production_library_candidates() -> tuple[Path, ...]:
    """Fixed CMake output locations, mirroring native_projection's contract."""
    source_root = Path(__file__).resolve().parents[4]
    return (
        source_root / "build" / "astra_analytical" /
        "build_congestion_aware" / "lib" / "libjoint_ledger_cabi.so",
        source_root / "build_congestion_aware" / "lib" /
        "libjoint_ledger_cabi.so",
    )


class _Counters(ctypes.Structure):
    _fields_ = [
        ("version", ctypes.c_uint64),
        ("event_watermark", ctypes.c_uint64),
        ("last_tick", ctypes.c_int64),
        ("last_delivery_sequence", ctypes.c_int64),
        ("full_load_count", ctypes.c_uint64),
        ("delta_load_count", ctypes.c_uint64),
        ("applied_event_count", ctypes.c_uint64),
        ("replayed_event_count", ctypes.c_uint64),
        ("pipeline_delivery_sequence", ctypes.c_int64),
        ("pipeline_watermark", ctypes.c_uint64),
        ("pipeline_state_version", ctypes.c_uint64),
        ("active_works", ctypes.c_uint64),
        ("parse_ns", ctypes.c_int64),
        ("stage_ns", ctypes.c_int64),
        ("commit_ns", ctypes.c_int64),
        ("last_snapshot_full", ctypes.c_int32),
    ]


class _ScanResult(ctypes.Structure):
    _fields_ = [
        ("member_start", ctypes.c_uint64),
        ("member_end", ctypes.c_uint64),
        ("value_start", ctypes.c_uint64),
        ("value_end", ctypes.c_uint64),
    ]


class NativeLedgerLibrary:
    """ctypes binding of JointLedgerCApi.h (ABI validated on load)."""

    def __init__(self, path: Path):
        self.path = path
        try:
            self.lib = ctypes.CDLL(str(path))
        except OSError as exc:
            raise NativeLedgerError(
                f"cannot load joint_ledger_cabi at {path}: {exc}") from exc
        lib = self.lib
        lib.jl_create.restype = ctypes.c_void_p
        lib.jl_destroy.argtypes = [ctypes.c_void_p]
        lib.jl_last_error.argtypes = [
            ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t]
        lib.jl_last_error.restype = ctypes.c_int
        lib.jl_scan_request.argtypes = [
            ctypes.c_char_p, ctypes.c_size_t, ctypes.POINTER(_ScanResult)]
        lib.jl_scan_request.restype = ctypes.c_int
        lib.jl_prepare.argtypes = [
            ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t,
            ctypes.c_uint64, ctypes.c_uint64, ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_uint64)]
        lib.jl_prepare.restype = ctypes.c_int
        lib.jl_commit.argtypes = [ctypes.c_void_p, ctypes.c_int32]
        lib.jl_commit.restype = ctypes.c_int
        lib.jl_discard_prepared.argtypes = [ctypes.c_void_p, ctypes.c_int32]
        lib.jl_discard_prepared.restype = ctypes.c_int
        lib.jl_counters.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(_Counters)]
        lib.jl_counters.restype = ctypes.c_int
        for name in ("jl_pricing_json", "jl_conservation_json",
                     "jl_works_export_json", "jl_pipeline_snapshot_json"):
            fn = getattr(lib, name)
            fn.argtypes = [
                ctypes.c_void_p, ctypes.c_char_p,
                ctypes.POINTER(ctypes.c_size_t)]
            fn.restype = ctypes.c_int
        # ABI sanity: create/destroy round-trip must succeed.
        handle = lib.jl_create()
        if not handle:
            raise NativeLedgerError(
                f"joint_ledger_cabi at {path} failed jl_create")
        lib.jl_destroy(ctypes.c_void_p(handle))

    # -- error helpers -------------------------------------------------
    def last_error(self, handle) -> str:
        buffer = ctypes.create_string_buffer(512)
        code = self.lib.jl_last_error(
            handle if handle else None,
            buffer, ctypes.sizeof(buffer))
        text = buffer.value.decode("utf-8", "replace")
        return text or f"unknown native error (code path {code})"

    def _json_call(self, name: str, handle) -> Any:
        fn = getattr(self.lib, name)
        size = ctypes.c_size_t(0)
        code = fn(handle, None, ctypes.byref(size))
        if code == 5:  # JL_ERR_RANGE: size query
            pass
        elif code != 0:
            raise NativeLedgerError(self.last_error(handle))
        buffer = ctypes.create_string_buffer(size.value)
        code = fn(handle, buffer, ctypes.byref(size))
        if code != 0:
            raise NativeLedgerError(self.last_error(handle))
        return json.loads(buffer.value.decode("utf-8"))


@lru_cache(maxsize=1)
def load_production_ledger_library() -> NativeLedgerLibrary:
    candidates = production_library_candidates()
    path = next((item for item in candidates if item.is_file()), None)
    if path is None:
        checked = ", ".join(str(item) for item in candidates)
        raise NativeLedgerError(
            "native joint dual-ledger receiver library is required "
            f"(libjoint_ledger_cabi.so); checked: {checked}")
    return NativeLedgerLibrary(path)


# ------------------------------------------------------------ raw scan --

def scan_request(raw: bytes) -> Tuple[bytes, bytes]:
    """Locate the top-level work_ledger member of one bridge request.

    Returns ``(outer_bytes, segment_bytes)``: the splice that removes the
    whole member (outer frame JSON, parsed by the caller with json.loads)
    and the member value's exact raw bytes (fed to the native receiver and
    the journal sidecar).  Raises NativeLedgerError on any malformed
    structure -- the fail-closed path mirrors the Python schema gates.
    """
    library = load_production_ledger_library()
    result = _ScanResult()
    code = library.lib.jl_scan_request(raw, len(raw), ctypes.byref(result))
    if code != 0:
        raise NativeLedgerError(library.last_error(None))
    outer = raw[:result.member_start] + raw[result.member_end:]
    segment = raw[result.value_start:result.value_end]
    return outer, segment


# ------------------------------------------------------------ receiver --

class _PipelinePricingView(Mapping):
    """MappingProxy-compatible view over the native pipeline pricing JSON."""

    def __init__(self, payload: Mapping[str, Any]):
        self._payload = dict(payload)

    def __getitem__(self, key):
        return self._payload[key]

    def __iter__(self):
        return iter(self._payload)

    def __len__(self):
        return len(self._payload)


class NativeDualLedgerReceiver:
    """Persistent dual-ledger receiver owned by the online scheduler.

    The Python scheduler holds this handle plus the small query results; all
    ledger state (works, descriptors, cursors, slots, credits, obligations,
    indexes, replay metadata) lives in the native library.
    """

    def __init__(self, *, library: Optional[NativeLedgerLibrary] = None):
        self._library = library or load_production_ledger_library()
        self._handle = self._library.lib.jl_create()
        if not self._handle:
            raise NativeLedgerError("jl_create failed")
        self._handle = ctypes.c_void_p(self._handle)
        self._pricing_cache: Optional[LedgerPricingSnapshot] = None
        self._pricing_cache_version: int = -1
        self._pipeline_pricing_cache: Optional[_PipelinePricingView] = None
        self._pipeline_pricing_cache_version: int = -1

    def close(self) -> None:
        if self._handle:
            self._library.lib.jl_destroy(self._handle)
            self._handle = None

    def __del__(self):  # best effort; scheduler owns the lifetime explicitly
        try:
            self.close()
        except Exception:
            pass

    # ------------------------------------------------------ delivery --

    def receive(self, segment: bytes, *, delivery_sequence: int,
                tick: int) -> Dict[str, Any]:
        """Prepare + atomically commit one work_ledger segment.

        Returns the delivery receipt facts (snapshot_kind, event_count,
        applied/replayed counters, event_watermark, state versions).  Any
        validation failure raises NativeLedgerError with the native row
        path and leaves both ledgers untouched.
        """
        prepared = ctypes.c_int32(-1)
        event_count = ctypes.c_uint64(0)
        seq_end = ctypes.c_uint64(0)
        result_version = ctypes.c_uint64(0)
        code = self._library.lib.jl_prepare(
            self._handle, segment, len(segment), int(delivery_sequence),
            int(tick), ctypes.byref(prepared), ctypes.byref(event_count),
            ctypes.byref(seq_end), ctypes.byref(result_version))
        if code != 0:
            raise NativeLedgerError(self._library.last_error(self._handle))
        code = self._library.lib.jl_commit(self._handle, prepared)
        if code != 0:
            raise NativeLedgerError(self._library.last_error(self._handle))
        counters = self.counters()
        return {
            "snapshot_kind": "full" if counters.last_snapshot_full else "delta",
            "delivery_sequence": int(delivery_sequence),
            "logical_event_count": int(event_count.value),
            "event_watermark": int(seq_end.value),
            "result_state_version": int(result_version.value),
            "applied_events": int(counters.applied_event_count),
            "replayed_events": int(counters.replayed_event_count),
            "parse_ns": int(counters.parse_ns),
            "stage_ns": int(counters.stage_ns),
            "commit_ns": int(counters.commit_ns),
        }

    def receive_payload_dict(self, payload: Mapping[str, Any]) -> Dict[str, Any]:
        """Dict-entry adaptation for embedded/offline callers (same core)."""
        if not isinstance(payload, Mapping):
            raise NativeLedgerError(
                "work_ledger payload must be a mapping for the dict entry")
        segment = json.dumps(
            payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        delivery = payload.get("delivery_sequence")
        tick = payload.get("tick")
        if not isinstance(delivery, int) or not isinstance(tick, int):
            raise NativeLedgerError(
                "work_ledger payload requires integer delivery_sequence/tick")
        return self.receive(segment, delivery_sequence=delivery, tick=tick)

    # ------------------------------------------------------ counters --

    def counters(self) -> _Counters:
        counters = _Counters()
        code = self._library.lib.jl_counters(
            self._handle, ctypes.byref(counters))
        if code != 0:
            raise NativeLedgerError(self._library.last_error(self._handle))
        return counters

    # Work-ledger attribute surface (online_scheduler_base / sh30 readers).
    @property
    def version(self) -> int:
        return int(self.counters().version)

    @property
    def event_watermark(self) -> int:
        return int(self.counters().event_watermark)

    @property
    def full_load_count(self) -> int:
        return int(self.counters().full_load_count)

    @property
    def delta_load_count(self) -> int:
        return int(self.counters().delta_load_count)

    @property
    def applied_event_count(self) -> int:
        return int(self.counters().applied_event_count)

    @property
    def replayed_event_count(self) -> int:
        return int(self.counters().replayed_event_count)

    @property
    def last_delivery_sequence(self) -> int:
        return int(self.counters().last_delivery_sequence)

    def active_work_count(self) -> int:
        return int(self.counters().active_works)

    def last_phase_ns(self) -> Dict[str, int]:
        counters = self.counters()
        return {
            "ledger_typed_parse_ns": int(counters.parse_ns),
            "ledger_prepare_ns": int(counters.stage_ns),
            "ledger_commit_ns": int(counters.commit_ns),
        }

    # --------------------------------------------------------- views --

    def pricing_snapshot(self) -> LedgerPricingSnapshot:
        """Reconstruct the small work-ledger pricing frontier DTO.

        Rows are the active leg works only (hbm:/pool:/noc:, issued or
        later) in projection insertion order -- identical membership and
        ordering contract to WorkLedger.pricing_snapshot.
        """
        counters = self.counters()
        if self._pricing_cache is not None and (
                self._pricing_cache_version == counters.version):
            return self._pricing_cache
        payload = self._library._json_call("jl_pricing_json", self._handle)
        rows = tuple(_projection_row(row)
                     for row in payload["work"]["pricing_rows"])
        snapshot = LedgerPricingSnapshot(
            pricing_rows=rows,
            committed_not_issued=int(payload["work"]["committed_not_issued"]))
        self._pricing_cache = snapshot
        self._pricing_cache_version = counters.version
        return snapshot

    def pipeline_pricing_snapshot(self) -> _PipelinePricingView:
        counters = self.counters()
        if self._pipeline_pricing_cache is not None and (
                self._pipeline_pricing_cache_version
                == counters.pipeline_state_version):
            return self._pipeline_pricing_cache
        payload = self._library._json_call("jl_pricing_json", self._handle)
        view = _PipelinePricingView(payload["pipeline"])
        self._pipeline_pricing_cache = view
        self._pipeline_pricing_cache_version = counters.pipeline_state_version
        return view

    # Pipeline attribute surface (sh30 snapshot / offline harness).
    @property
    def pipeline_initialized(self) -> bool:
        return self.counters().pipeline_delivery_sequence >= 0

    @property
    def pipeline_state_version(self) -> int:
        return int(self.counters().pipeline_state_version)

    @property
    def pipeline_watermark(self) -> int:
        return int(self.counters().pipeline_watermark)

    # ------------------------------------------------- diagnostics --

    def conservation_report(self) -> Dict[str, Any]:
        return self._library._json_call(
            "jl_conservation_json", self._handle)

    def export_state(self) -> Dict[str, Any]:
        """WorkLedger.export_state-equivalent (offline harness/digests)."""
        return self._library._json_call("jl_works_export_json", self._handle)

    def pipeline_snapshot_state(self) -> Dict[str, Any]:
        """PipelineLedger.remaining_view-equivalent (offline harness)."""
        return self._library._json_call(
            "jl_pipeline_snapshot_json", self._handle)

    def works_summary(self) -> Dict[str, Dict[str, Any]]:
        export = self.export_state()
        summary: Dict[str, Dict[str, Any]] = {}
        for block in export.get("nodes", []):
            for node in block.get("nodes", ()):
                summary[node["work_id"]] = {
                    "state": "completed" if node.get("finished")
                    else "issued" if node.get("issued") else "committed",
                    "dependencies": list(node.get("dependencies", ())),
                }
        return summary

    def iter_active_works(self) -> Iterator[CommittedWork]:
        """Reconstruct CommittedWork objects from the native export.

        Only used by diagnostic/test consumers without a pricing projection
        (production always carries the snapshot); mirrors
        WorkLedger.iter_active_works over the native accepted state.
        """
        export = self.export_state()
        yield from _committed_works_from_export(export)


class NativePipelineLedgerFacade:
    """Pipeline-side attribute surface over the shared native receiver.

    Mirrors the PipelineLedgerV4 read face consumed by the scheduler:
    ``initialized`` + ``pricing_snapshot()`` (epoch/delivery_sequence/
    event_watermark/state_version/tick/active_resources).
    """

    def __init__(self, receiver: NativeDualLedgerReceiver):
        self._receiver = receiver

    @property
    def initialized(self) -> bool:
        return self._receiver.pipeline_initialized

    @property
    def state_version(self) -> int:
        return self._receiver.pipeline_state_version

    @property
    def watermark(self) -> int:
        return self._receiver.pipeline_watermark

    @property
    def delivery_sequence(self) -> int:
        return int(self._receiver.counters().pipeline_delivery_sequence)

    def pricing_snapshot(self):
        return self._receiver.pipeline_pricing_snapshot()

    def remaining_view(self) -> Dict[str, Any]:
        """PipelineLedger.remaining_view-equivalent for offline harness."""
        return self._receiver.pipeline_snapshot_state()

    def snapshot_state(self) -> Dict[str, Any]:
        """PipelineLedger.snapshot_state-equivalent (same view)."""
        return self._receiver.pipeline_snapshot_state()


# ------------------------------------------------------- DTO rebuild --

def _opt_amount(row: Any) -> WorkAmount:
    if row is None:
        return WorkAmount()
    return WorkAmount(
        ops=None if row.get("ops") is None else float(row["ops"]),
        bytes=None if row.get("bytes") is None else float(row["bytes"]),
        latency_ns=None if row.get("latency_ns") is None
        else float(row["latency_ns"]),
    )


def _projection_row(row: Mapping[str, Any]) -> LedgerProjectionRow:
    identity = row["identity"]
    return LedgerProjectionRow(
        identity=WorkIdentity(
            request_id=identity["request_id"],
            stage=identity["stage"],
            generation=int(identity["generation"]),
            rank=int(identity["rank"]),
            node_id=int(identity["node_id"]),
            operation=identity.get("operation", ""),
        ),
        state=row["state"],
        total=_opt_amount(row["total"]),
        served=_opt_amount(row["served"]),
        remaining=_opt_amount(row["remaining"]),
        progress_source=row["progress_source"],
        service_unit=row["service_unit"],
        dependencies=tuple(row.get("dependencies", ())),
        release_conditions=tuple(row.get("release_conditions", ())),
        route=(tuple(row["route"]) if row.get("route") is not None else None),
        weight=int(row.get("weight", 0)),
        leg=row.get("leg"),
        committed_tick=row.get("committed_tick"),
        issued_tick=row.get("issued_tick"),
        completed_tick=row.get("completed_tick"),
        acked_tick=row.get("acked_tick"),
        released_tick=row.get("released_tick"),
        as_of_tick=row.get("as_of_tick"),
    )


def _committed_works_from_export(export: Mapping[str, Any]):
    """Best-effort CommittedWork reconstruction from export_state rows."""
    for block in export.get("nodes", []):
        rank = int(block["rank"])
        for node in block.get("nodes", ()):
            identity = WorkIdentity(
                request_id=node["request_id"], stage=node["stage"],
                generation=int(node["generation"]), rank=rank,
                node_id=int(node["node_id"]),
                operation=node.get("kind", "invalid"))
            work = CommittedWork(
                identity=identity,
                state="completed" if node.get("finished")
                else "issued" if node.get("issued") else "committed",
                total=WorkAmount(
                    ops=float(node.get("num_ops") or 0),
                    bytes=float(node.get("bytes") or 0),
                    latency_ns=float(node.get("runtime_ns") or 0)),
                progress_source=node.get("progress_source", "stale"),
                service_unit="ops" if identity.operation == "compute"
                else "bytes",
                dependencies=tuple(node.get("dependencies", ())),
            )
            work.remaining = work.total
            if node.get("finished"):
                work.remaining = WorkAmount()
                work.served = work.total
            yield work
