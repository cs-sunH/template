"""Process-style operation capture dispatch."""

from __future__ import annotations

from typing import Any

from .access import (
    _base_attrs,
    _ensure_shape_pair_compatible,
    _gemm_dimensions,
    _is_buffer_like,
    _normalize_access,
    _validate_reduce_shapes,
    _validate_vector_broadcast,
)
from ..ir.nodes import OpAction
from ..language import ProcessStyleCall


# Public dispatch

def capture_process_call(
    process_call: ProcessStyleCall,
    *,
    source_line: int | None,
    conditions: tuple[Any, ...],
) -> OpAction:
    op_name = process_call.op
    if op_name == "copy":
        return _capture_copy(process_call, source_line, conditions)
    if op_name == "gemm":
        return _capture_gemm(process_call, source_line, conditions)
    if op_name == "send":
        return _capture_send(process_call, source_line, conditions)
    if op_name == "recv":
        return _capture_recv(process_call, source_line, conditions)
    if op_name in {"reduce", "cumsum"}:
        return _capture_reduce(process_call, source_line, conditions)
    return _capture_vector(process_call, source_line, conditions)


# Operation capture

def _capture_copy(process_call: ProcessStyleCall, source_line: int | None, conditions: tuple[Any, ...]) -> OpAction:
    if len(process_call.args) != 1 or process_call.out is None:
        raise ValueError("copy capture requires one source and one destination.")
    src_access = _normalize_access(process_call.args[0], "read")
    dst_access = _normalize_access(process_call.out, "write")
    if src_access.slice_shape is None and dst_access.slice_shape is not None:
        src_access = _normalize_access(process_call.args[0], "read", expected_shape=dst_access.slice_shape)
    if dst_access.slice_shape is None and src_access.slice_shape is not None:
        dst_access = _normalize_access(process_call.out, "write", expected_shape=src_access.slice_shape)
    _ensure_shape_pair_compatible(src_access.slice_shape, dst_access.slice_shape, "copy")
    return OpAction(
        action_kind="tensor_copy",
        input_accesses=[src_access],
        output_accesses=[dst_access],
        attrs=_base_attrs("copy", source_line, conditions),
    )


def _capture_gemm(process_call: ProcessStyleCall, source_line: int | None, conditions: tuple[Any, ...]) -> OpAction:
    if len(process_call.args) != 2 or process_call.out is None:
        raise ValueError("gemm capture requires two inputs and one output.")
    input_a = _normalize_access(process_call.args[0], "read")
    input_b = _normalize_access(process_call.args[1], "read")
    output = _normalize_access(process_call.out, "write")
    attrs = _base_attrs("gemm", source_line, conditions)
    attrs.update(process_call.attrs)
    attrs.update(_gemm_dimensions(input_a.slice_shape, input_b.slice_shape, attrs))
    return OpAction(
        action_kind="matrix_compute",
        input_accesses=[input_a, input_b],
        output_accesses=[output],
        attrs=attrs,
    )


def _capture_vector(process_call: ProcessStyleCall, source_line: int | None, conditions: tuple[Any, ...]) -> OpAction:
    output_access = _normalize_access(process_call.out, "write") if process_call.out is not None else None
    input_accesses = [_normalize_access(arg, "read") for arg in process_call.args if _is_buffer_like(arg)]
    if output_access is None:
        raise ValueError(f"{process_call.op} process-style capture requires an output buffer.")
    _validate_vector_broadcast(input_accesses, output_access, process_call.op)
    attrs = _base_attrs(process_call.op, source_line, conditions)
    attrs.update(process_call.attrs)
    return OpAction(
        action_kind="vector_compute",
        input_accesses=input_accesses,
        output_accesses=[output_access],
        attrs=attrs,
    )


def _capture_reduce(process_call: ProcessStyleCall, source_line: int | None, conditions: tuple[Any, ...]) -> OpAction:
    if len(process_call.args) != 1 or process_call.out is None:
        raise ValueError(f"{process_call.op} capture requires one input and one output.")
    input_access = _normalize_access(process_call.args[0], "read")
    output_access = _normalize_access(process_call.out, "write")
    attrs = _base_attrs(process_call.op, source_line, conditions)
    attrs.update(process_call.attrs)
    _validate_reduce_shapes(input_access.slice_shape, output_access.slice_shape, attrs, process_call.op)
    return OpAction(
        action_kind="vector_compute",
        input_accesses=[input_access],
        output_accesses=[output_access],
        attrs=attrs,
    )


def _capture_send(process_call: ProcessStyleCall, source_line: int | None, conditions: tuple[Any, ...]) -> OpAction:
    if len(process_call.args) != 3:
        raise ValueError("send capture requires src_core, dst_core, and a payload.")
    src_core, dst_core, payload = process_call.args
    payload_access = _normalize_access(payload, "read")
    attrs = _base_attrs("send", source_line, conditions)
    attrs.update({"src_core": src_core, "dst_core": dst_core})
    return OpAction(action_kind="noc_send", input_accesses=[payload_access], attrs=attrs)


def _capture_recv(process_call: ProcessStyleCall, source_line: int | None, conditions: tuple[Any, ...]) -> OpAction:
    if len(process_call.args) != 2 or process_call.out is None:
        raise ValueError("recv capture requires src_core, dst_core, and a payload.")
    src_core, dst_core = process_call.args
    payload_access = _normalize_access(process_call.out, "write")
    attrs = _base_attrs("recv", source_line, conditions)
    attrs.update({"src_core": src_core, "dst_core": dst_core})
    return OpAction(action_kind="noc_recv", output_accesses=[payload_access], attrs=attrs)

__all__ = ["capture_process_call"]
