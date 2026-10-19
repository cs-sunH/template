"""Typed ctypes facade for the native joint projection C ABI.

The serializers below keep the Python dataclasses as the source of truth.  A
native context receives one frozen ProjectionInput snapshot; each project call
passes only the bounded candidate WorkItems and per-call options.  DiePlan
descriptors are serialized as typed JSON, never expanded into per-tile Python
objects.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import json
import math
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
import time
from typing import Any, Mapping, Optional, Sequence

from joint.resource_projection import (
    BackgroundComparison,
    ProjectionInput,
    ProjectionResult,
    STATUS_BLOCKED_NO_PROGRESS,
    STATUS_DEGRADED,
    STATUS_MODEL_UNAVAILABLE,
    STATUS_OK,
    STATUS_WAIT_PROGRESS,
)


class NativeProjectionError(RuntimeError):
    """C ABI load, serialization, parse, or native validation failure."""


def production_library_candidates() -> tuple[Path, ...]:
    """Return the fixed CMake output locations checked by production.

    This is deliberately not an environment/configuration selector.  An
    explicit path is accepted only by the test/evidence injection API below;
    normal online scheduling resolves the repository's conventional build
    output and fails closed when it is absent.
    """
    # Conventional build outputs resolve relative to the repository this
    # module lives in (exec doc 20.1: the acceptance driver builds run
    # copies at the same conventional location inside the copy), so an
    # out-of-workspace run copy finds its own build instead of assuming
    # the workspace directory layout. The evidence-root candidate stays
    # workspace-relative as a live-tree convenience.
    source_root = Path(__file__).resolve().parents[4]
    repo_root = Path(__file__).resolve().parents[6]
    return (
        source_root / "build" / "astra_analytical" /
        "build_congestion_aware" / "lib" / "libjoint_projection.so",
        source_root / "build_congestion_aware" / "lib" /
        "libjoint_projection.so",
        repo_root / "packages" / "cpp-core" / "build" /
        "congestion_aware" / "lib" / "libjoint_projection.so",
    )


@lru_cache(maxsize=1)
def load_production_projection_library() -> "NativeProjectionLibrary":
    """Load the one production ABI or raise with every checked path."""
    candidates = production_library_candidates()
    path = next((item for item in candidates if item.is_file()), None)
    if path is None:
        checked = ", ".join(str(item) for item in candidates)
        raise NativeProjectionError(
            "native joint projection library is required (ABI 1); checked: "
            + checked)
    try:
        return NativeProjectionLibrary(path)
    except NativeProjectionError as exc:
        raise NativeProjectionError(
            f"production projection library {path} failed ABI 1 validation: {exc}") from exc


_DISCLOSURE_KEYS = (
    "unsupported_reason", "logical_event_count", "physical_step_count",
    "reference_cache_hits", "reference_cache_misses",
)
_RESULT_KEYS = (
    "status", "predicted_service_done_ns", "predicted_settle_done_ns",
    "reason", "dependency_attribution", "focus_bytes", "bytes_served_total",
    "event_count", "resource_count", "work_count",
    "closure_resource_count", "closure_work_count", "approximations",
    "fallback", "background_comparisons",
)
_COARSE_DISCLOSURE_KEYS = (
    "coarse_projection_profile", "live_world_import_supported",
    "coarse_runtime_counters_by_domain", "coarse_plan_evidence",
)
_STATUSES = {
    STATUS_OK, STATUS_WAIT_PROGRESS, STATUS_BLOCKED_NO_PROGRESS,
    STATUS_DEGRADED, STATUS_MODEL_UNAVAILABLE,
}


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"),
            ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise NativeProjectionError(
            "projection payload is not finite JSON") from exc


def _resource_wire(resource) -> dict:
    return {
        "key": resource.key,
        "kind": resource.kind,
        "capacity_per_ns": float(resource.capacity_per_ns),
    }


def _lease_wire(lease) -> dict:
    return {"gate_id": lease.gate_id, "units": lease.units}


def _task_wire(leg) -> dict:
    die_plan = leg.die_plan
    if die_plan is not None:
        serializer = getattr(die_plan, "to_wire", None)
        if not callable(serializer):
            raise NativeProjectionError(
                f"die plan on {leg.leg_id!r} has no typed to_wire() serializer")
        die_plan = serializer()
    order = leg.order
    if order is not None and (
            type(order) is not int or not -(1 << 63) <= order < (1 << 63)):
        raise NativeProjectionError(
            f"TaskSpec.order for {leg.leg_id!r} must be signed int64 or null")
    registration_ordinal = leg.registration_ordinal
    if registration_ordinal is not None and (
            type(registration_ordinal) is not int
            or not 0 <= registration_ordinal <= (1 << 64) - 1):
        raise NativeProjectionError(
            f"TaskSpec.registration_ordinal for {leg.leg_id!r} must be uint64 or null")
    wire = {
        "leg_id": leg.leg_id,
        "kind": leg.kind,
        "route": list(leg.route),
        "bytes_amount": leg.bytes_amount,
        "duration_ns": float(leg.duration_ns),
        "base_ns": float(leg.base_ns),
        "group": leg.group,
        "latency_ns": leg.latency_ns,
        "ready_at_ns": leg.ready_at_ns,
        "deps": list(leg.deps),
        "acquires": [_lease_wire(item) for item in leg.acquires],
        "releases": [_lease_wire(item) for item in leg.releases],
        "signals_barrier": leg.signals_barrier,
        "remaining_bytes": leg.remaining_bytes,
        "remaining_duration_ns": leg.remaining_duration_ns,
        "in_service": bool(leg.in_service),
        "order": order,
        "registration_ordinal": registration_ordinal,
        "die_plan": die_plan,
    }
    endpoint = getattr(leg, "coarse_endpoint", None)
    if endpoint is not None:
        wire["coarse_endpoint"] = json.loads(_canonical_json_bytes(dict(endpoint)))
    return wire


def _work_wire(work) -> dict:
    return {
        "work_id": work.work_id,
        "legs": [_task_wire(leg) for leg in work.legs],
        "ready_at_ns": work.ready_at_ns,
        "depends_on_work": list(work.depends_on_work),
        "is_candidate": bool(work.is_candidate),
        "service_leg_ids": (None if work.service_leg_ids is None else
                             list(work.service_leg_ids)),
    }


def projection_input_to_wire(input_data: ProjectionInput) -> dict:
    """Serialize one complete immutable old-world input without tile expansion."""
    wire = {
        "now_ns": input_data.now_ns,
        "resources": [_resource_wire(item) for item in input_data.resources],
        "background_works": [
            _work_wire(item) for item in input_data.background_works],
        "gates": [{
            "gate_id": item.gate_id,
            "kind": item.kind,
            "capacity_units": item.capacity_units,
            "initial_occupied": item.initial_occupied,
        } for item in input_data.gates],
        "barriers": [{
            "barrier_id": item.barrier_id,
            "signallers": list(item.signallers),
        } for item in input_data.barriers],
        "efficiency": {
            "eta": dict(input_data.efficiency.eta),
            "gamma": dict(input_data.efficiency.gamma),
        },
        "pipelines": [{
            "pipeline_id": item.pipeline_id,
            "fill_resource": item.fill_resource,
            "block_bytes": item.block_bytes,
            "num_blocks": item.num_blocks,
            "consume_duration_ns": float(item.consume_duration_ns),
            "buffer_gate_id": item.buffer_gate_id,
            "buffer_slots": item.buffer_slots,
            "startup_ns": item.startup_ns,
        } for item in input_data.pipelines],
        "forecast_arrival_placeholders_ns": list(
            input_data.forecast_arrival_placeholders_ns),
        "focus_work_id": input_data.focus_work_id,
    }
    if getattr(input_data, "forecast_start_index", None) is not None:
        wire["forecast_start_index"] = int(input_data.forecast_start_index)
        # 共享底座可能为空背景（锚与否取决于逐调用 overlay 的 extras），
        # focus 为占位符——由原生 overlay 应用时重算，创建期不校验。
        wire["defer_focus_validation"] = True
    if input_data.coarse_die_domains:
        wire["coarse_die_domains"] = [{
            "resource_key": domain.resource_key,
            "hardware_config": json.loads(_canonical_json_bytes(
                dict(domain.hardware_config))),
        } for domain in input_data.coarse_die_domains]
    if input_data.coarse_cancel_events:
        wire["coarse_cancel_events"] = [{
            "time_ns": event.time_ns,
            "leg_id": event.leg_id,
        } for event in input_data.coarse_cancel_events]
    return wire


def _candidate_wire(candidates: Sequence) -> list:
    result = []
    ids = set()
    for item in candidates:
        if not item.is_candidate:
            raise NativeProjectionError(
                f"candidate work {item.work_id!r} is not marked is_candidate")
        if item.work_id in ids:
            raise NativeProjectionError(
                f"duplicate candidate work {item.work_id!r}")
        ids.add(item.work_id)
        result.append(_work_wire(item))
    return result


def _overlay_wire(overlay: Mapping[str, Any]) -> dict:
    """Serialize one per-call private overlay (shared-snapshot form).

    Fields mirror the candidate-dependent parts of the legacy per-candidate
    ProjectionInput: candidate resources (already reduced to the additive
    setdefault subset), gates, barriers, extra background works, and the
    empty-background anchor flag.
    """
    if not isinstance(overlay, Mapping):
        raise NativeProjectionError("private overlay must be a mapping")
    expected = {
        "resources", "extra_background_works", "gates", "barriers",
        "append_background_anchor"}
    if set(overlay) - expected:
        raise NativeProjectionError(
            "private overlay has unknown keys: "
            f"{sorted(set(overlay) - expected)}")
    return {
        "resources": [_resource_wire(item) for item in overlay.get(
            "resources", ())],
        "extra_background_works": [
            _work_wire(item) for item in overlay.get(
                "extra_background_works", ())],
        "gates": [{
            "gate_id": item.gate_id,
            "kind": item.kind,
            "capacity_units": item.capacity_units,
            "initial_occupied": item.initial_occupied,
        } for item in overlay.get("gates", ())],
        "barriers": [{
            "barrier_id": item.barrier_id,
            "signallers": list(item.signallers),
        } for item in overlay.get("barriers", ())],
        "append_background_anchor": bool(
            overlay.get("append_background_anchor", False)),
    }


def projection_options_to_wire(
        params, *, horizon_ns: Optional[float] = None,
        include_background_reference: bool = True,
        analytic_segments: bool = True, drain_background: bool = True) -> dict:
    """Build per-call controls, keeping the run contract immutable."""
    if horizon_ns is not None and (
            isinstance(horizon_ns, bool)
            or not isinstance(horizon_ns, (int, float))
            or not math.isfinite(float(horizon_ns)) or horizon_ns < 0):
        raise NativeProjectionError("horizon_ns must be null or finite >= 0")
    return {
        "projection_event_budget": int(params.projection_event_budget),
        "closure_resource_budget": int(params.closure_resource_budget),
        "horizon_ns": horizon_ns,
        "include_background_reference": bool(include_background_reference),
        "analytic_segments": bool(analytic_segments),
        "drain_background": bool(drain_background),
    }


def _int(value, path: str, *, nullable: bool = False) -> Optional[int]:
    if nullable and value is None:
        return None
    if type(value) is not int or value < 0:
        raise NativeProjectionError(f"native result {path} must be a non-negative integer")
    return value


def _signed_int(value, path: str, *, nullable: bool = False) -> Optional[int]:
    """Signed int64 field (e.g. background comparison delay).

    The native emitter computes ``delay_ns`` as ``after - before`` in signed
    int64 and the legacy ``BackgroundComparison.delay_ns`` property carries
    the same signed difference: service quantization can legitimately make
    the with-candidate world finish a background work one tick earlier.
    Endpoints stay non-negative; only the difference is signed.
    """
    if nullable and value is None:
        return None
    if type(value) is not int or not -(1 << 63) <= value < (1 << 63):
        raise NativeProjectionError(
            f"native result {path} must be a signed int64")
    return value


def projection_result_from_wire(wire: Mapping[str, Any], *,
                                elapsed_ns: int = 0) -> ProjectionResult:
    """Strictly map deterministic native output to the existing result API."""
    if not isinstance(wire, Mapping):
        raise NativeProjectionError("native projection result must be an object")
    expected = set(_RESULT_KEYS) | set(_DISCLOSURE_KEYS)
    coarse = set(_COARSE_DISCLOSURE_KEYS) & set(wire)
    if coarse:
        if coarse != set(_COARSE_DISCLOSURE_KEYS):
            raise NativeProjectionError("native coarse disclosure must be complete")
        if (wire["coarse_projection_profile"] != "fresh_integer_local_domains_v1"
                or wire["live_world_import_supported"] is not False
                or not isinstance(wire["coarse_runtime_counters_by_domain"], Mapping)
                or not isinstance(wire["coarse_plan_evidence"], Mapping)):
            raise NativeProjectionError("native coarse disclosure has an unsupported profile/state")
        expected |= coarse
    if set(wire) != expected:
        raise NativeProjectionError(
            "native result keys differ from the v1 contract "
            f"(missing={sorted(expected - set(wire))}, "
            f"unknown={sorted(set(wire) - expected)})")
    status = wire["status"]
    if status not in _STATUSES:
        raise NativeProjectionError(f"native result has unknown status {status!r}")
    for field in ("predicted_service_done_ns", "predicted_settle_done_ns"):
        _int(wire[field], field, nullable=True)
    for field in ("focus_bytes", "bytes_served_total", "event_count",
                  "resource_count", "work_count", "closure_resource_count",
                  "closure_work_count"):
        _int(wire[field], field)
    for field in ("reason",):
        if not isinstance(wire[field], str):
            raise NativeProjectionError(f"native result {field} must be a string")
    if not isinstance(wire["dependency_attribution"], list) or any(
            not isinstance(item, str) for item in wire["dependency_attribution"]):
        raise NativeProjectionError("native dependency_attribution must be a string array")
    if not isinstance(wire["approximations"], list) or any(
            not isinstance(item, str) for item in wire["approximations"]):
        raise NativeProjectionError("native approximations must be a string array")
    if wire["fallback"] is not None and not isinstance(wire["fallback"], str):
        raise NativeProjectionError("native fallback must be string or null")
    if not isinstance(wire["background_comparisons"], list):
        raise NativeProjectionError("native background_comparisons must be an array")
    comparisons = []
    for index, item in enumerate(wire["background_comparisons"]):
        if not isinstance(item, Mapping) or set(item) != {
                "work_id", "done_without_candidate_ns",
                "done_with_candidate_ns", "delay_ns"}:
            raise NativeProjectionError(
                f"native background_comparisons[{index}] has invalid fields")
        if not isinstance(item["work_id"], str):
            raise NativeProjectionError("native comparison work_id must be a string")
        before = _int(item["done_without_candidate_ns"], "comparison.before", nullable=True)
        after = _int(item["done_with_candidate_ns"], "comparison.after", nullable=True)
        delay = _signed_int(item["delay_ns"], "comparison.delay", nullable=True)
        comparison = BackgroundComparison(item["work_id"], before, after)
        if comparison.delay_ns != delay:
            raise NativeProjectionError("native comparison delay disagrees with its endpoints")
        comparisons.append(comparison)
    for field in ("logical_event_count", "physical_step_count",
                  "reference_cache_hits", "reference_cache_misses"):
        _int(wire[field], field)
    unsupported = wire["unsupported_reason"]
    if unsupported is not None and not isinstance(unsupported, str):
        raise NativeProjectionError("native unsupported_reason must be string or null")
    disclosure = {key: wire[key] for key in _DISCLOSURE_KEYS}
    disclosure.update({key: wire[key] for key in coarse})
    return ProjectionResult(
        status=status,
        predicted_service_done_ns=wire["predicted_service_done_ns"],
        predicted_settle_done_ns=wire["predicted_settle_done_ns"],
        reason=wire["reason"],
        dependency_attribution=tuple(wire["dependency_attribution"]),
        focus_bytes=wire["focus_bytes"],
        bytes_served_total=wire["bytes_served_total"],
        event_count=wire["event_count"],
        resource_count=wire["resource_count"],
        work_count=wire["work_count"],
        closure_resource_count=wire["closure_resource_count"],
        closure_work_count=wire["closure_work_count"],
        approximations=tuple(wire["approximations"]),
        fallback=wire["fallback"],
        background_comparisons=tuple(comparisons),
        candidate_world_wall_ns=max(0, int(elapsed_ns)),
        native_disclosure=disclosure,
    )


class NativeProjectionLibrary:
    """Loaded v1 C ABI. A context is immutable; calls isolate candidates."""

    def __init__(self, library_path: str | Path) -> None:
        path = Path(library_path)
        if not path.is_file():
            raise NativeProjectionError(f"native projection library is missing: {path}")
        try:
            self._lib = ctypes.CDLL(str(path))
        except OSError as exc:
            raise NativeProjectionError(f"cannot load native projection library {path}: {exc}") from exc
        self._bind_abi()

    def _bind_abi(self) -> None:
        lib = self._lib
        lib.jproj_abi_version.argtypes = []
        lib.jproj_abi_version.restype = ctypes.c_uint32
        if lib.jproj_abi_version() != 1:
            raise NativeProjectionError("unsupported joint_projection C ABI version")
        lib.jproj_create.argtypes = [ctypes.c_char_p, ctypes.c_size_t,
                                     ctypes.POINTER(ctypes.c_void_p)]
        lib.jproj_create.restype = ctypes.c_int
        lib.jproj_project.argtypes = [
            ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t,
            ctypes.c_char_p, ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_size_t)]
        lib.jproj_project.restype = ctypes.c_int
        lib.jproj_world_operation.argtypes = [
            ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_size_t)]
        lib.jproj_world_operation.restype = ctypes.c_int
        lib.jproj_free_result.argtypes = [ctypes.c_void_p]
        lib.jproj_free_result.restype = None
        lib.jproj_destroy.argtypes = [ctypes.c_void_p]
        lib.jproj_destroy.restype = None
        lib.jproj_last_error.argtypes = [ctypes.c_void_p]
        lib.jproj_last_error.restype = ctypes.c_char_p

    def _error(self, context, prefix: str) -> NativeProjectionError:
        raw = self._lib.jproj_last_error(context)
        detail = raw.decode("utf-8", errors="replace") if raw else "unknown native error"
        return NativeProjectionError(f"{prefix}: {detail}")

    def create_context(self, input_data: ProjectionInput,
                       kernel_contract: Optional[Mapping[str, Any]] = None
                       ) -> "NativeProjectionContext":
        envelope = {"projection_input": projection_input_to_wire(input_data)}
        if kernel_contract is not None:
            envelope["kernel_contract"] = dict(kernel_contract)
        payload = _canonical_json_bytes(envelope)
        context = ctypes.c_void_p()
        code = self._lib.jproj_create(payload, len(payload), ctypes.byref(context))
        if code != 0 or not context.value:
            raise self._error(None, "jproj_create failed")
        return NativeProjectionContext(self, context)


class NativeProjectionContext:
    """One immutable native snapshot, reusable for independent candidate calls.

    v5 shared-snapshot form: the context freezes the decision-level shared
    background once; per-call candidate payloads carry the bounded candidate
    WorkItems plus a ``private_overlay`` (candidate-declared resources, gates,
    barriers and §7.2 inherited-obligation background works) that the native
    side merges exactly like the legacy per-candidate snapshot did.  The
    context additionally memoizes background-reference runs keyed by the
    overlay content and run options, so candidates sharing those inputs run
    the reference world once.
    """

    def __init__(self, library: NativeProjectionLibrary,
                 context: ctypes.c_void_p) -> None:
        self._library = library
        self._context = context
        self._closed = False

    def close(self) -> None:
        if not self._closed:
            self._library._lib.jproj_destroy(self._context)
            self._closed = True

    def __del__(self) -> None:  # leak safety net for non-arbitrate callers
        try:
            self.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()

    def project(self, candidates: Sequence, params, *,
                horizon_ns: Optional[float] = None,
                include_background_reference: bool = True,
                analytic_segments: bool = True,
                drain_background: bool = True,
                private_overlay: Optional[Mapping[str, Any]] = None
                ) -> ProjectionResult:
        if self._closed:
            raise NativeProjectionError("native projection context is closed")
        payload = {"candidates": _candidate_wire(candidates)}
        if private_overlay is not None:
            payload["private_overlay"] = _overlay_wire(private_overlay)
        candidate_payload = _canonical_json_bytes(payload)
        options_payload = _canonical_json_bytes(projection_options_to_wire(
            params, horizon_ns=horizon_ns,
            include_background_reference=include_background_reference,
            analytic_segments=analytic_segments,
            drain_background=drain_background))
        result_ptr = ctypes.c_void_p()
        result_len = ctypes.c_size_t()
        started = time.perf_counter_ns()
        code = self._library._lib.jproj_project(
            self._context, candidate_payload, len(candidate_payload),
            options_payload, len(options_payload),
            ctypes.byref(result_ptr), ctypes.byref(result_len))
        elapsed = time.perf_counter_ns() - started
        try:
            if code < 0:
                raise self._library._error(self._context, "jproj_project failed")
            if not result_ptr.value:
                raise NativeProjectionError("native projection returned no result buffer")
            payload = ctypes.string_at(result_ptr, result_len.value)
            try:
                wire = json.loads(payload.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise NativeProjectionError("native projection returned invalid JSON") from exc
            result = projection_result_from_wire(wire, elapsed_ns=elapsed)
            if code == 1 and result.status not in (
                    STATUS_DEGRADED, STATUS_MODEL_UNAVAILABLE):
                raise NativeProjectionError(
                    "C ABI returned a budget/model status inconsistent with the result")
            if code == 0 and result.status in (STATUS_DEGRADED,
                                               STATUS_MODEL_UNAVAILABLE):
                raise NativeProjectionError(
                    "C ABI returned success for a degraded/unavailable result")
            return result
        finally:
            if result_ptr.value:
                self._library._lib.jproj_free_result(result_ptr)

    def world_operation(self, request: Mapping[str, Any]) -> dict:
        """Forward one stateful-world operation through the dedicated C ABI.

        The session belongs to this context, while ``project`` always builds
        isolated query worlds.  Keeping the two entry points separate avoids
        accidental mutation of the frozen snapshot or candidate result.
        """
        if self._closed:
            raise NativeProjectionError("native projection context is closed")
        if not isinstance(request, Mapping):
            raise NativeProjectionError("world operation request must be an object")
        if not isinstance(request.get("operation"), str):
            raise NativeProjectionError("world operation needs string operation")
        payload = _canonical_json_bytes(dict(request))
        result_ptr = ctypes.c_void_p()
        result_len = ctypes.c_size_t()
        code = self._library._lib.jproj_world_operation(
            self._context, payload, len(payload),
            ctypes.byref(result_ptr), ctypes.byref(result_len))
        try:
            if code != 0:
                raise self._library._error(
                    self._context, "jproj_world_operation failed")
            if not result_ptr.value:
                raise NativeProjectionError(
                    "native world operation returned no result buffer")
            raw = ctypes.string_at(result_ptr, result_len.value)
            try:
                result = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise NativeProjectionError(
                    "native world operation returned invalid JSON") from exc
            if not isinstance(result, dict):
                raise NativeProjectionError(
                    "native world operation result must be an object")
            return result
        finally:
            if result_ptr.value:
                self._library._lib.jproj_free_result(result_ptr)


class NativeProjectionLegView:
    """Compatibility view of one leg in the native mutable world."""

    __slots__ = (
        "spec", "work_id", "state", "remaining_bytes",
        "remaining_duration", "completion", "cancelled_at", "order_key",
        "die_actual_bytes_total", "die_service_units_total")

    def __init__(self, wire: Mapping[str, Any], spec=None) -> None:
        self.spec = spec
        self.work_id = wire["work_id"]
        self.state = wire["state"]
        self.remaining_bytes = float(wire["remaining_bytes"])
        self.remaining_duration = float(wire["remaining_duration_ns"])
        self.completion = wire["completion_ns"]
        self.cancelled_at = wire.get("cancelled_at_ns")
        self.order_key = tuple(wire["order_key"])
        self.die_actual_bytes_total = int(wire["die_actual_bytes_total"])
        self.die_service_units_total = float(
            wire["die_service_units_total"])


class NativeProjectionWorld:
    """Stateful native replacement surface for ``ProjectionWorld``.

    This adapter deliberately forwards every state transition, including the
    conservation-test hooks, to the native session.  It does not duplicate
    the Python event loop.  Pass the run's parsed kernel contract so the
    mutable world and isolated projection calls share the same frozen ABI
    version and semantic prediction configuration digest.
    """

    def __init__(self, input_data: ProjectionInput, *,
                 library: NativeProjectionLibrary,
                 kernel_contract: Optional[Mapping[str, Any]] = None,
                 efficiency=None) -> None:
        if not isinstance(library, NativeProjectionLibrary):
            raise NativeProjectionError(
                "NativeProjectionWorld requires a NativeProjectionLibrary")
        if efficiency is not None and efficiency != input_data.efficiency:
            raise NativeProjectionError(
                "native world efficiency override must equal snapshot efficiency")
        if kernel_contract is None:
            from joint.prediction_config import (
                load_prediction_config, make_pipeline_kernel_contract)
            kernel_contract = make_pipeline_kernel_contract(
                load_prediction_config())
        self.input = input_data
        self.efficiency = input_data.efficiency if efficiency is None else efficiency
        self._context = library.create_context(input_data, kernel_contract)
        self._owned_context = True
        self._conservation_check = True
        self._analytic_mode = False
        self._now_ns = float(input_data.now_ns)
        self.resources = {item.key: item for item in input_data.resources}
        self._spec_by_leg = {
            leg.leg_id: leg
            for work in input_data.background_works for leg in work.legs}
        self._add_block_pipeline_specs(input_data)
        self._sync(self._context.world_operation({"operation": "inspect"}))

    @classmethod
    def from_context(cls, input_data: ProjectionInput,
                     context: NativeProjectionContext, *,
                     efficiency=None):
        """Use a shared immutable native snapshot/session owned by caller."""
        if not isinstance(context, NativeProjectionContext):
            raise NativeProjectionError(
                "NativeProjectionWorld.from_context needs a native context")
        instance = cls.__new__(cls)
        instance.input = input_data
        instance.efficiency = input_data.efficiency if efficiency is None else efficiency
        if instance.efficiency != input_data.efficiency:
            raise NativeProjectionError(
                "native world efficiency override must equal snapshot efficiency")
        instance._context = context
        instance._owned_context = False
        instance._conservation_check = True
        instance._analytic_mode = False
        instance._now_ns = float(input_data.now_ns)
        instance.resources = {item.key: item for item in input_data.resources}
        instance._spec_by_leg = {
            leg.leg_id: leg
            for work in input_data.background_works for leg in work.legs}
        instance._add_block_pipeline_specs(input_data)
        instance._sync(context.world_operation({"operation": "inspect"}))
        return instance

    def _add_block_pipeline_specs(self, input_data) -> None:
        from joint.resource_projection import (
            GateLease, TASK_KIND_FIXED, TASK_KIND_STREAM, TaskSpec)
        for pipeline in input_data.pipelines:
            work_id = f"pipeline:{pipeline.pipeline_id}"
            for index in range(pipeline.num_blocks):
                fill_id = f"{work_id}#fill{index}"
                consume_id = f"{work_id}#consume{index}"
                self._spec_by_leg[fill_id] = TaskSpec(
                    leg_id=fill_id, kind=TASK_KIND_STREAM,
                    route=(pipeline.fill_resource,),
                    bytes_amount=pipeline.block_bytes,
                    latency_ns=(pipeline.startup_ns if index == 0 else 0),
                    deps=((f"{work_id}#fill{index - 1}",) if index else ()),
                    acquires=(GateLease(pipeline.buffer_gate_id),))
                self._spec_by_leg[consume_id] = TaskSpec(
                    leg_id=consume_id, kind=TASK_KIND_FIXED,
                    duration_ns=pipeline.consume_duration_ns,
                    deps=(fill_id,) + ((f"{work_id}#consume{index - 1}",)
                                       if index else ()),
                    releases=(GateLease(pipeline.buffer_gate_id),))

    def close(self) -> None:
        if self._owned_context:
            self._context.close()
            self._owned_context = False

    def _sync(self, wire: Mapping[str, Any]) -> None:
        required = {
            "now_ns", "status", "events_processed", "integration_steps",
            "work_done_at", "work_remaining", "legs", "cost_disclosure",
        }
        if not isinstance(wire, Mapping) or not required.issubset(wire):
            raise NativeProjectionError(
                "native world inspect is missing required state fields")
        self._now_ns = float(wire["now_ns"])
        self.status = wire["status"]
        self.events_processed = int(wire["events_processed"])
        self.integration_steps = int(wire["integration_steps"])
        self.work_done_at = dict(wire["work_done_at"])
        self.work_remaining = {
            key: int(value) for key, value in wire["work_remaining"].items()}
        self.legs = {
            leg_id: NativeProjectionLegView(
                leg_wire, self._spec_by_leg.get(leg_id))
            for leg_id, leg_wire in wire["legs"].items()}
        self.work_legs = {}
        for leg_id, leg in self.legs.items():
            self.work_legs.setdefault(leg.work_id, []).append(leg)
        from joint.resource_projection import GateSpec, BarrierSpec
        self.gates = {}
        gate_wire = wire.get("gates", {})
        for gate in self.input.gates:
            value = gate_wire.get(gate.gate_id)
            if value is None:
                # Older ABI builds omit runtime gate detail.  Preserve the
                # immutable spec view without claiming any progress update.
                self.gates[gate.gate_id] = SimpleNamespace(
                    spec=gate, occupied=gate.initial_occupied,
                    max_occupied=gate.initial_occupied, waiters=[])
                continue
            spec = GateSpec(
                gate_id=gate.gate_id, kind=value["kind"],
                capacity_units=int(value["capacity"]),
                initial_occupied=int(value["initial_occupied"]))
            self.gates[gate.gate_id] = SimpleNamespace(
                spec=spec, occupied=int(value["occupied"]),
                max_occupied=int(value["max_occupied"]),
                waiters=[self.legs[leg_id]
                         for leg_id in value["waiting_leg_ids"]
                         if leg_id in self.legs])
        self.barriers = {}
        for barrier_id, value in wire.get("barriers", {}).items():
            # Native inspect uses the BarrierSpec vocabulary.  Accept the
            # older Python-view aliases as well so a rolling ABI build still
            # reconstructs the same immutable signaller set.
            signallers = value.get("signallers", value.get("expected"))
            if signallers is None:
                raise NativeProjectionError(
                    f"native barrier {barrier_id!r} is missing signallers")
            spec = BarrierSpec(barrier_id, tuple(signallers))
            passed_at = value.get("passed_at_ns", value.get("passed_at"))
            has_passed = value.get("has_passed", passed_at is not None)
            if not isinstance(has_passed, bool):
                raise NativeProjectionError(
                    f"native barrier {barrier_id!r} has invalid has_passed")
            if has_passed and passed_at is None:
                raise NativeProjectionError(
                    f"native barrier {barrier_id!r} passed without a timestamp")
            self.barriers[barrier_id] = SimpleNamespace(
                spec=spec, arrived=set(value["arrived"]),
                passed_at=passed_at if has_passed else None)
        self._active_legs = [
            leg for leg in self.legs.values() if leg.state == "active"]
        self.bytes_served_total = float(
            wire["cost_disclosure"].get("bytes_served_total", 0))
        self._native_cost_disclosure = dict(wire["cost_disclosure"])

    def _operation(self, request: Mapping[str, Any]) -> dict:
        try:
            result = self._context.world_operation(request)
        except NativeProjectionError as exc:
            if request.get("operation") == "assert_conservation":
                from joint.resource_projection import ProjectionConservationError
                raise ProjectionConservationError(str(exc)) from exc
            raise
        if {"now_ns", "work_done_at", "legs", "cost_disclosure"}.issubset(result):
            self._sync(result)
        return result

    def _step(self) -> bool:
        return bool(self._operation({"operation": "step"}).get("progressed"))

    @property
    def now_ns(self) -> float:
        return self._now_ns

    @now_ns.setter
    def now_ns(self, value: float) -> None:
        # The old private conservation tests inject a clock-only violation by
        # assigning ``world.now_ns`` directly.  Preserve that test hook while
        # forwarding the mutation to the native session instead of keeping a
        # Python-only clock that could diverge from native state.
        if not hasattr(self, "_context") or not hasattr(self, "_now_ns"):
            self._now_ns = float(value)
        else:
            self.set_clock(value)

    def run(self, *, horizon_ns: Optional[float] = None,
            event_budget: Optional[int] = None,
            analytic_segments: bool = False,
            drain_background: bool = True) -> str:
        request = {
            "operation": "run",
            "horizon_ns": horizon_ns,
            "event_budget": event_budget,
            "analytic_segments": bool(analytic_segments),
            "drain_background": bool(drain_background),
        }
        self._analytic_mode = bool(analytic_segments)
        return self._operation(request)["status"]

    def reset(self, *, event_budget: Optional[int] = None,
              conservation_check: bool = True) -> None:
        self._conservation_check = bool(conservation_check)
        request = {"operation": "reset",
                   "conservation_check": bool(conservation_check)}
        if event_budget is not None:
            request["event_budget"] = event_budget
        self._operation(request)

    def _integrate(self, target_ns: float) -> None:
        self._operation({"operation": "integrate", "target_ns": target_ns})

    def _assert_conservation(self, pre_remaining: float,
                             served_total: float, rates: Mapping[str, float],
                             interval_start_ns: float) -> None:
        # A list of pairs preserves the active-leg insertion order that the
        # reference Python mapping carried into its ordered floating sum.
        ordered_rates = [[leg_id, rate] for leg_id, rate in rates.items()]
        self._operation({
            "operation": "assert_conservation",
            "pre_remaining": pre_remaining,
            "served_total": served_total,
            "rates": ordered_rates,
            "interval_start_ns": interval_start_ns,
        })

    def set_clock(self, now_ns: float) -> None:
        self._operation({"operation": "set_clock", "now_ns": now_ns})

    def work_completion(self, work_id: str):
        return self.work_done_at.get(work_id)

    def service_completion(self, work_id: str):
        for work in self.input.background_works:
            if work.work_id != work_id:
                continue
            service_ids = (work.service_leg_ids if work.service_leg_ids is not None
                           else tuple(leg.leg_id for leg in work.legs))
            times = [self.legs[leg_id].completion for leg_id in service_ids]
            if any(value is None for value in times):
                return None
            return max(times) if times else self.work_done_at.get(work_id)
        return None

    def cost_disclosure(self) -> dict:
        return dict(self._native_cost_disclosure)

    def export_snapshot(self) -> dict:
        """Return the complete native wire snapshot for this mutable world."""
        return self._operation({"operation": "export_snapshot"})

    def import_snapshot(self, snapshot: Mapping[str, Any]) -> dict:
        """Atomically restore a native snapshot and refresh façade controls.

        The native session validates the frozen graph and complete runtime
        state before publishing the import.  Python only mirrors the two
        session controls exposed by this compatibility façade.
        """
        result = self._operation({
            "operation": "import_snapshot", "snapshot": snapshot})
        self._conservation_check = snapshot["controls"]["conservation_check"]
        self._analytic_mode = snapshot["state"]["analytic_enabled"]
        return result
