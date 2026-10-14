"""Small ctypes client for the coarse DIE compile-only workspace envelope.

This module deliberately contains no workspace model.  It forwards the exact
resolved hardware JSON bytes and typed task JSON to the DIE C ABI, then checks
the versioned response and its identities.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Mapping, Optional, Union


Pathish = Union[str, os.PathLike]
_LIBRARY_ENV = "DIE_COARSE_CABI_PATH"
_LIBRARY_NAME = "libcoarse_die_cabi.so"
_UINT64_MAX = (1 << 64) - 1
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

class WorkspaceEnvelopeError(RuntimeError):
    """The C ABI, input, or envelope response violated its contract."""


@dataclass(frozen=True)
class WorkspaceEnvelope:
    schema_version: int
    workspace_envelope_schema_version: int
    candidate_count: int
    max_hbm_workspace_peak_bytes: int
    hardware_fingerprint: str
    normalized_task_fingerprint: str
    op_kind: str
    phase: str
    normalized_task: dict[str, Any]


@dataclass(frozen=True)
class WorkspaceCapacityBound:
    """Candidate-independent lifetime bound; not an executability result."""

    capacity_model_revision: str
    hbm_workspace_upper_bound_bytes: int
    hardware_fingerprint: str
    normalized_task_fingerprint: str
    op_kind: str
    phase: str
    normalized_task: dict[str, Any]


def _default_repo_root() -> Path:
    # Match config_resolver.py's root, which is sh_test_mesh/../.
    return Path(__file__).resolve().parents[4]


def coarse_cabi_candidates(repo_root: Optional[Pathish] = None) -> tuple[Path, Path]:
    """Return the same ordered repository-local candidates as config_resolver."""
    root = Path(repo_root) if repo_root is not None else _default_repo_root()
    build = root / "build" / "astra_analytical" / "build_congestion_aware"
    return (
        build / "lib" / _LIBRARY_NAME,
        build / "bin" / _LIBRARY_NAME,
    )


def discover_coarse_cabi_library(
    library_path: Optional[Pathish] = None,
    *,
    repo_root: Optional[Pathish] = None,
) -> Path:
    """Resolve an explicit library or the config_resolver discovery sequence.

    An explicit path is authoritative.  Otherwise the environment path is
    considered first, followed by the repository's lib and bin build paths.
    A selected but incompatible library fails closed when symbols are bound.
    """
    if library_path is not None:
        selected = Path(library_path).expanduser()
        if not selected.is_file():
            raise WorkspaceEnvelopeError(
                f"coarse DIE C ABI library does not exist: {selected}"
            )
        return selected.resolve()

    candidates: list[Path] = []
    env_value = os.environ.get(_LIBRARY_ENV)
    if env_value:
        candidates.append(Path(env_value).expanduser())
    local_candidates = coarse_cabi_candidates(repo_root)
    candidates.extend(local_candidates)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise WorkspaceEnvelopeError(
        f"{_LIBRARY_NAME} not found; build the coarse_die_cabi target or set "
        f"{_LIBRARY_ENV}; repository candidates: "
        + " -> ".join(str(path) for path in local_candidates)
    )


def _require_u64(value: Any, path: str) -> int:
    if type(value) is not int or value < 0 or value > _UINT64_MAX:
        raise WorkspaceEnvelopeError(
            f"{path} must be an exact uint64 integer (bool and float are rejected)"
        )
    return value


def _validate_json_object_keys(value: Any, path: str = "task") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise WorkspaceEnvelopeError(f"{path} object keys must be strings")
            _validate_json_object_keys(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _validate_json_object_keys(child, f"{path}[{index}]")


def _validate_u64_fields(
    value: Any, keys: frozenset[str], path: str
) -> None:
    if not isinstance(value, Mapping):
        return
    for key in keys:
        if key in value:
            _require_u64(value[key], f"{path}.{key}")


def _validate_span_u64s(value: Any, path: str, *, legacy: bool) -> None:
    if not isinstance(value, (list, tuple)):
        return
    if legacy:
        span_keys = frozenset(
            {"first_index", "q_tokens", "kv_tokens", "q_start_pos", "kv_start_pos"}
        )
        run_keys = frozenset({"q_tokens", "kv_first", "kv_stride", "count"})
    else:
        span_keys = frozenset(
            {
                "batch", "count", "q_len", "q_tokens", "kv_len", "kv_tokens",
                "q_start_pos", "query_start", "kv_start_pos", "kv_start",
            }
        )
        run_keys = frozenset(
            {
                "q_len", "q_tokens", "kv_len", "kv_tokens", "kv_first",
                "kv_stride", "q_start_pos", "kv_start_pos", "count",
            }
        )
    for index, span in enumerate(value):
        span_path = f"{path}[{index}]"
        _validate_u64_fields(span, span_keys, span_path)
        if isinstance(span, Mapping) and "run" in span:
            _validate_u64_fields(span["run"], run_keys, f"{span_path}.run")


def _validate_task_u64s(value: Any, path: str = "task") -> None:
    """Reject Python bool/float values only at actual task uint64 fields."""
    _validate_json_object_keys(value, path)
    if not isinstance(value, Mapping):
        return

    _validate_u64_fields(value, frozenset({"layer_instances", "weight_passes"}), path)
    legacy = "die_op_spec" in value or (
        "model_dims" in value and "schema_version" in value
    )
    if "die_op_spec" in value and isinstance(value["die_op_spec"], Mapping):
        spec = value["die_op_spec"]
        spec_path = f"{path}.die_op_spec"
    else:
        spec = value
        spec_path = path

    if legacy:
        _validate_u64_fields(
            spec,
            frozenset({"schema_version", "layer_count", "weight_passes"}),
            spec_path,
        )
        _validate_u64_fields(
            spec.get("model_dims"),
            frozenset(
                {
                    "hidden", "ffn", "vocab", "num_q_heads", "num_kv_heads",
                    "head_dim",
                }
            ),
            f"{spec_path}.model_dims",
        )
        _validate_u64_fields(
            spec.get("tp_slice"),
            frozenset(
                {
                    "tensor_parallel", "tensor_parallel_rank", "q_heads_local",
                    "kv_heads_local", "ffn_local", "vocab_local",
                }
            ),
            f"{spec_path}.tp_slice",
        )
        _validate_span_u64s(spec.get("spans"), f"{spec_path}.spans", legacy=True)
        return

    _validate_u64_fields(
        value.get("tensor_shapes"),
        frozenset(
            {
                "m", "M", "n", "N", "k", "K", "hidden", "h", "q_heads",
                "num_q_heads", "q_heads_local", "kv_heads", "num_kv_heads",
                "kv_heads_local", "head_dim", "ffn", "ffn_local", "vocab",
                "vocab_local",
            }
        ),
        f"{path}.tensor_shapes",
    )
    _validate_u64_fields(
        value.get("tp_slice"),
        frozenset(
            {
                "tensor_parallel", "tensor_parallel_rank", "q_heads_local",
                "kv_heads_local", "ffn_local", "vocab_local",
            }
        ),
        f"{path}.tp_slice",
    )
    _validate_span_u64s(value.get("spans"), f"{path}.spans", legacy=False)


def _last_error(library: Any) -> str:
    raw = library.die_coarse_last_error_v2()
    if not raw:
        return ""
    return raw.decode("utf-8", errors="replace")


def _read_owned_json(library: Any, pointer: Any, description: str) -> Any:
    """Copy and parse one owned C string, freeing it on every parse path."""
    if not pointer:
        raise WorkspaceEnvelopeError(f"{description} returned a null result")
    try:
        raw = ctypes.string_at(pointer)
        text = raw.decode("utf-8", errors="strict")
        return json.loads(text)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise WorkspaceEnvelopeError(
            f"{description} returned invalid UTF-8 JSON: {error}"
        ) from error
    finally:
        library.die_coarse_free_result_v2(pointer)


def _bind_library(library: Any, library_path: Path) -> None:
    required = (
        "die_coarse_abi_version",
        "die_coarse_create_model_v2",
        "die_coarse_destroy_model_v2",
        "die_coarse_free_result_v2",
        "die_coarse_hardware_summary_v2",
        "die_coarse_last_error_v2",
        "die_coarse_workspace_envelope_v2",
    )
    missing = [name for name in required if not hasattr(library, name)]
    if missing:
        raise WorkspaceEnvelopeError(
            f"{library_path} is missing required coarse DIE C ABI symbol(s): "
            + ", ".join(missing)
        )

    library.die_coarse_abi_version.argtypes = []
    library.die_coarse_abi_version.restype = ctypes.c_uint32
    library.die_coarse_create_model_v2.argtypes = [
        ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_uint64),
    ]
    library.die_coarse_create_model_v2.restype = ctypes.c_int
    library.die_coarse_destroy_model_v2.argtypes = [ctypes.c_uint64]
    library.die_coarse_destroy_model_v2.restype = ctypes.c_int
    library.die_coarse_free_result_v2.argtypes = [ctypes.c_void_p]
    library.die_coarse_free_result_v2.restype = None
    library.die_coarse_hardware_summary_v2.argtypes = [ctypes.c_uint64]
    library.die_coarse_hardware_summary_v2.restype = ctypes.c_void_p
    library.die_coarse_last_error_v2.argtypes = []
    library.die_coarse_last_error_v2.restype = ctypes.c_char_p
    library.die_coarse_workspace_envelope_v2.argtypes = [
        ctypes.c_uint64,
        ctypes.c_char_p,
    ]
    library.die_coarse_workspace_envelope_v2.restype = ctypes.c_void_p

    abi_version = library.die_coarse_abi_version()
    if abi_version != 2:
        raise WorkspaceEnvelopeError(
            f"{library_path} reports coarse DIE ABI version {abi_version}; expected 2"
        )


def _require_object(value: Any, description: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise WorkspaceEnvelopeError(f"{description} must be a JSON object")
    return value


def _require_fingerprint(value: Any, description: str) -> str:
    if not isinstance(value, str) or not value:
        raise WorkspaceEnvelopeError(f"{description} must be a non-empty string")
    return value


def _canonical_task_fingerprint(task: dict[str, Any]) -> str:
    # nlohmann::json's default object ordering and compact UTF-8 dump match
    # sort_keys/separators here for the normalized typed task schema.
    payload = json.dumps(
        task,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class DieWorkspaceEnvelopeClient:
    """Own one DIE model and query compile-only envelopes for typed tasks."""

    def __init__(
        self,
        config_path: Pathish,
        library_path: Optional[Pathish] = None,
    ) -> None:
        self.config_path = Path(config_path).expanduser().resolve()
        try:
            hardware_bytes = self.config_path.read_bytes()
        except OSError as error:
            raise WorkspaceEnvelopeError(
                f"cannot read resolved hardware config {self.config_path}: {error}"
            ) from error
        try:
            hardware_text = hardware_bytes.decode("utf-8", errors="strict")
            hardware_json = json.loads(hardware_text)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise WorkspaceEnvelopeError(
                f"resolved hardware config must be exact UTF-8 JSON: {error}"
            ) from error
        if not isinstance(hardware_json, dict):
            raise WorkspaceEnvelopeError("resolved hardware config must be a JSON object")

        self.library_path = discover_coarse_cabi_library(library_path)
        try:
            self._library = ctypes.CDLL(str(self.library_path))
        except OSError as error:
            raise WorkspaceEnvelopeError(
                f"cannot load coarse DIE C ABI library {self.library_path}: {error}"
            ) from error
        _bind_library(self._library, self.library_path)
        self._model_handle: Optional[int] = None
        self._hardware_fingerprint: Optional[str] = None

        model_out = ctypes.c_uint64(0)
        status = self._library.die_coarse_create_model_v2(
            hardware_bytes, ctypes.byref(model_out)
        )
        if status != 0 or model_out.value == 0:
            detail = _last_error(self._library)
            raise WorkspaceEnvelopeError(
                "die_coarse_create_model_v2 failed"
                + (f": {detail}" if detail else f" with status {status}")
            )
        self._model_handle = int(model_out.value)

        try:
            summary_pointer = self._library.die_coarse_hardware_summary_v2(
                self._model_handle
            )
            if not summary_pointer:
                detail = _last_error(self._library)
                raise WorkspaceEnvelopeError(
                    "die_coarse_hardware_summary_v2 failed"
                    + (f": {detail}" if detail else " with a null result")
                )
            summary = _require_object(
                _read_owned_json(
                    self._library,
                    summary_pointer,
                    "die_coarse_hardware_summary_v2",
                ),
                "hardware summary",
            )
            schema_version = _require_u64(
                summary.get("schema_version"), "hardware_summary.schema_version"
            )
            if schema_version != 2:
                raise WorkspaceEnvelopeError(
                    f"hardware summary schema_version is {schema_version}; expected 2"
                )
            self._hardware_fingerprint = _require_fingerprint(
                summary.get("hardware_fingerprint"),
                "hardware_summary.hardware_fingerprint",
            )
        except Exception:
            self.close()
            raise

    @property
    def hardware_fingerprint(self) -> str:
        self._require_open()
        assert self._hardware_fingerprint is not None
        return self._hardware_fingerprint

    @property
    def model_handle(self) -> int:
        """Opaque C ABI handle, exposed for diagnostics and focused tests."""
        return self._require_open()

    def _require_open(self) -> int:
        if self._model_handle is None:
            raise WorkspaceEnvelopeError("workspace envelope client is closed")
        return self._model_handle

    def query(self, task: Mapping[str, Any]) -> WorkspaceEnvelope:
        model_handle = self._require_open()
        if not isinstance(task, Mapping):
            raise WorkspaceEnvelopeError("task must be a JSON object mapping")
        _validate_task_u64s(task)
        try:
            task_bytes = json.dumps(
                task,
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        except (TypeError, ValueError, UnicodeEncodeError) as error:
            raise WorkspaceEnvelopeError(f"task is not valid UTF-8 JSON: {error}") from error

        pointer = self._library.die_coarse_workspace_envelope_v2(
            model_handle, task_bytes
        )
        if not pointer:
            detail = _last_error(self._library)
            raise WorkspaceEnvelopeError(
                "die_coarse_workspace_envelope_v2 failed"
                + (f": {detail}" if detail else " without last_error detail")
            )
        raw_envelope = _require_object(
            _read_owned_json(
                self._library,
                pointer,
                "die_coarse_workspace_envelope_v2",
            ),
            "workspace envelope",
        )
        return self._validate_envelope(raw_envelope)

    def query_capacity_bound(self, task: Mapping[str, Any]) -> WorkspaceCapacityBound:
        """Ask C++ for all allowed lifetimes before SRAM feasibility filtering."""
        model_handle = self._require_open()
        if not isinstance(task, Mapping):
            raise WorkspaceEnvelopeError("task must be a JSON object mapping")
        _validate_task_u64s(task)
        symbol = "die_coarse_workspace_capacity_bound_v2"
        if not hasattr(self._library, symbol):
            raise WorkspaceEnvelopeError(f"coarse DIE library lacks {symbol}")
        query = getattr(self._library, symbol)
        query.argtypes = [ctypes.c_uint64, ctypes.c_char_p]
        query.restype = ctypes.c_void_p
        try:
            encoded = json.dumps(task, ensure_ascii=False, separators=(",", ":"),
                                 allow_nan=False).encode("utf-8")
        except (TypeError, ValueError, UnicodeEncodeError) as error:
            raise WorkspaceEnvelopeError(f"task is not valid UTF-8 JSON: {error}") from error
        pointer = query(model_handle, encoded)
        if not pointer:
            raise WorkspaceEnvelopeError(f"{symbol} failed: {_last_error(self._library)}")
        raw = _require_object(_read_owned_json(self._library, pointer, symbol),
                              "workspace capacity bound")
        if (_require_u64(raw.get("schema_version"), "schema_version") != 2 or
                _require_u64(raw.get("workspace_capacity_bound_schema_version"),
                             "workspace_capacity_bound_schema_version") != 1):
            raise WorkspaceEnvelopeError("unsupported workspace capacity bound schema")
        if (raw.get("capacity_model_revision") != "workspace_lifetimes_v1" or
                raw.get("capacity_bound_kind") !=
                "all_template_lifetimes_without_sram_filter"):
            raise WorkspaceEnvelopeError("unsupported workspace capacity lifetime model")
        peak = _require_u64(raw.get("hbm_workspace_upper_bound_bytes"),
                            "hbm_workspace_upper_bound_bytes")
        hardware = _require_fingerprint(raw.get("hardware_fingerprint"),
                                        "hardware_fingerprint")
        if hardware != self.hardware_fingerprint:
            raise WorkspaceEnvelopeError("capacity bound hardware identity mismatch")
        normalized = _require_object(raw.get("normalized_task"), "normalized_task")
        _validate_task_u64s(normalized, "normalized_task")
        fingerprint = _require_fingerprint(raw.get("normalized_task_fingerprint"),
                                           "normalized_task_fingerprint")
        if fingerprint != _canonical_task_fingerprint(normalized):
            raise WorkspaceEnvelopeError("capacity bound normalized task identity mismatch")
        op, phase = raw.get("op_kind"), raw.get("phase")
        if (normalized.get("schema") != "coarse_die_task_v2" or
                not isinstance(op, str) or not isinstance(phase, str) or
                normalized.get("op_kind") != op or normalized.get("phase") != phase):
            raise WorkspaceEnvelopeError("capacity bound operation/phase identity mismatch")
        return WorkspaceCapacityBound("workspace_lifetimes_v1", peak, hardware,
                                      fingerprint, op, phase, normalized)

    def _validate_envelope(self, raw: dict[str, Any]) -> WorkspaceEnvelope:
        schema_version = _require_u64(raw.get("schema_version"), "schema_version")
        envelope_schema_version = _require_u64(
            raw.get("workspace_envelope_schema_version"),
            "workspace_envelope_schema_version",
        )
        if schema_version != 2 or envelope_schema_version != 1:
            raise WorkspaceEnvelopeError(
                "unsupported workspace envelope schema: "
                f"schema_version={schema_version}, "
                f"workspace_envelope_schema_version={envelope_schema_version}"
            )

        candidate_count = _require_u64(raw.get("candidate_count"), "candidate_count")
        if candidate_count < 1 or candidate_count > 256:
            raise WorkspaceEnvelopeError(
                "candidate_count must be in the compiler's range [1, 256]"
            )
        peak_bytes = _require_u64(
            raw.get("max_hbm_workspace_peak_bytes"),
            "max_hbm_workspace_peak_bytes",
        )
        hardware_fingerprint = _require_fingerprint(
            raw.get("hardware_fingerprint"), "hardware_fingerprint"
        )
        if hardware_fingerprint != self._hardware_fingerprint:
            raise WorkspaceEnvelopeError(
                "workspace envelope hardware_fingerprint does not match the "
                "loaded model's hardware summary"
            )

        normalized_task = _require_object(raw.get("normalized_task"), "normalized_task")
        _validate_task_u64s(normalized_task, "normalized_task")
        if normalized_task.get("schema") != "coarse_die_task_v2":
            raise WorkspaceEnvelopeError(
                "normalized_task.schema must be coarse_die_task_v2"
            )
        normalized_op = normalized_task.get("op_kind")
        normalized_phase = normalized_task.get("phase")
        op_kind = raw.get("op_kind")
        phase = raw.get("phase")
        if not isinstance(normalized_op, str) or not isinstance(op_kind, str):
            raise WorkspaceEnvelopeError("op_kind must be a string in both identities")
        if not isinstance(normalized_phase, str) or not isinstance(phase, str):
            raise WorkspaceEnvelopeError("phase must be a string in both identities")
        if normalized_op != op_kind or normalized_phase != phase:
            raise WorkspaceEnvelopeError(
                "workspace envelope op_kind/phase disagree with normalized_task"
            )

        task_fingerprint = _require_fingerprint(
            raw.get("normalized_task_fingerprint"),
            "normalized_task_fingerprint",
        )
        expected_task_fingerprint = _canonical_task_fingerprint(normalized_task)
        if not _SHA256_RE.fullmatch(task_fingerprint) or task_fingerprint != expected_task_fingerprint:
            raise WorkspaceEnvelopeError(
                "normalized_task_fingerprint does not match the canonical "
                "normalized_task JSON"
            )

        return WorkspaceEnvelope(
            schema_version=schema_version,
            workspace_envelope_schema_version=envelope_schema_version,
            candidate_count=candidate_count,
            max_hbm_workspace_peak_bytes=peak_bytes,
            hardware_fingerprint=hardware_fingerprint,
            normalized_task_fingerprint=task_fingerprint,
            op_kind=op_kind,
            phase=phase,
            normalized_task=normalized_task,
        )

    def close(self) -> None:
        handle = self._model_handle
        if handle is None:
            return
        # Mark closed even if destroy reports a stale/bad handle, preventing a
        # second destructor call against a possibly released C ABI object.
        self._model_handle = None
        status = self._library.die_coarse_destroy_model_v2(handle)
        if status != 0:
            detail = _last_error(self._library)
            raise WorkspaceEnvelopeError(
                "die_coarse_destroy_model_v2 failed"
                + (f": {detail}" if detail else f" with status {status}")
            )

    def __enter__(self) -> "DieWorkspaceEnvelopeClient":
        self._require_open()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        try:
            self.close()
        except Exception as close_error:
            if exc is None:
                raise
            if hasattr(exc, "add_note"):
                exc.add_note(f"workspace envelope model cleanup also failed: {close_error}")
        return False

