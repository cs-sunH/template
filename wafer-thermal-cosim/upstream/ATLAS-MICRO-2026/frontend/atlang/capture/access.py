"""Tensor and buffer access normalization for captured process calls."""

from __future__ import annotations

import math
from typing import Any

from ..ir.dtype import dtype_byte_size
from ..ir.nodes import TensorAccess
from ..language import Buffer, BufferRegion, Tensor, TensorRegion


# Access normalization

def _normalize_access(value: Any, access_mode: str, expected_shape: tuple[Any, ...] | None = None) -> TensorAccess:
    if not _is_buffer_like(value):
        raise ValueError(f"Expected a tensor or buffer value, got {value!r}.")
    base, indices = _base_and_indices(value)
    name = getattr(base, "name", None)
    if not name:
        raise ValueError("Tensor/buffer access requires a source variable name.")
    base_shape = getattr(base, "shape", None)
    base_strides = _base_strides(base)
    index_expressions, slice_shape, slice_strides = _normalize_indices(indices, base_shape, base_strides)
    if slice_shape is None and expected_shape is not None:
        slice_shape = tuple(expected_shape)
        slice_strides = _point_region_strides(base_strides, len(slice_shape))
    byte_count = _byte_count(slice_shape, getattr(base, "dtype", None))
    return TensorAccess(
        tensor_name=name,
        access_mode=access_mode,
        index_expressions=index_expressions,
        slice_shape=slice_shape,
        slice_strides=slice_strides,
        byte_count=byte_count,
        attrs={"dtype": getattr(base, "dtype", None), "memory_space": _memory_space(base)},
    )


def _base_and_indices(value: Any) -> tuple[Any, tuple[Any, ...]]:
    if isinstance(value, (TensorRegion, BufferRegion)):
        base, parent_indices = _base_and_indices(value.base)
        return base, parent_indices + tuple(value.indices)
    return value, ()


def _normalize_indices(
    indices: tuple[Any, ...],
    base_shape: tuple[Any, ...] | None,
    base_strides: tuple[Any, ...] | None,
) -> tuple[tuple[Any, ...], tuple[Any, ...] | None, tuple[Any, ...] | None]:
    rank = len(base_shape or ())
    if not indices:
        if base_shape is None:
            return (), None, None
        return tuple(0 for _ in base_shape), tuple(base_shape), base_strides or _contiguous_strides(tuple(base_shape))
    index_expressions: list[Any] = []
    slice_shape: list[Any] = []
    slice_strides: list[Any] = []
    has_explicit_slice = False
    for axis, index in enumerate(indices):
        axis_stride = None if base_strides is None or axis >= len(base_strides) else base_strides[axis]
        axis_extent = None if base_shape is None or axis >= len(base_shape) else base_shape[axis]
        if isinstance(index, slice):
            has_explicit_slice = True
            start = 0 if index.start is None else index.start
            stop = axis_extent if index.stop is None else index.stop
            step = 1 if index.step is None else index.step
            index_expressions.append(start)
            slice_shape.append(_slice_extent(start, stop, step))
            slice_strides.append(None if axis_stride is None else axis_stride * step)
        else:
            index_expressions.append(index)
    if base_shape is not None and len(indices) < rank:
        for axis in range(len(indices), rank):
            axis_stride = None if base_strides is None else base_strides[axis]
            index_expressions.append(0)
            slice_shape.append(base_shape[axis])
            slice_strides.append(axis_stride)
            has_explicit_slice = True
    if not has_explicit_slice:
        return tuple(index_expressions), None, None
    return tuple(index_expressions), tuple(slice_shape), tuple(slice_strides)


# Validation and derived attrs

def _base_attrs(op_name: str, source_line: int | None, conditions: tuple[Any, ...]) -> dict[str, Any]:
    return {"op_name": op_name, "source_line": source_line, "conditions": conditions}


def _gemm_dimensions(
    input_a_shape: tuple[Any, ...] | None,
    input_b_shape: tuple[Any, ...] | None,
    attrs: dict[str, Any],
) -> dict[str, Any]:
    if input_a_shape is None or input_b_shape is None:
        return {}
    if len(input_a_shape) != 2 or len(input_b_shape) != 2:
        raise ValueError("gemm inputs must be rank-2 regions.")
    transpose_a = bool(attrs.get("transpose_A", False))
    transpose_b = bool(attrs.get("transpose_B", False))
    effective_a = (input_a_shape[1], input_a_shape[0]) if transpose_a else input_a_shape
    effective_b = (input_b_shape[1], input_b_shape[0]) if transpose_b else input_b_shape
    _ensure_shape_pair_compatible((effective_a[1],), (effective_b[0],), "gemm K dimension")
    return {"M": effective_a[0], "K": effective_a[1], "N": effective_b[1]}


def _validate_vector_broadcast(input_accesses: list[TensorAccess], output_access: TensorAccess, op_name: str) -> None:
    output_shape = output_access.slice_shape
    if output_shape is None:
        return
    for input_access in input_accesses:
        input_shape = input_access.slice_shape
        if input_shape is None:
            continue
        if not _can_broadcast(input_shape, output_shape):
            raise ValueError(f"{op_name} input shape {input_shape} cannot broadcast to output shape {output_shape}.")


def _validate_reduce_shapes(
    input_shape: tuple[Any, ...] | None,
    output_shape: tuple[Any, ...] | None,
    attrs: dict[str, Any],
    op_name: str,
) -> None:
    if input_shape is None or output_shape is None:
        return
    dim = int(attrs.get("dim", 0))
    if dim < 0:
        dim += len(input_shape)
    if dim < 0 or dim >= len(input_shape):
        raise ValueError(f"{op_name} dim {attrs.get('dim')} is out of range for shape {input_shape}.")
    attrs["reduce_dim"] = dim
    if op_name == "cumsum":
        _ensure_shape_pair_compatible(input_shape, output_shape, "cumsum")
        return
    reduced_shape = tuple(axis for index, axis in enumerate(input_shape) if index != dim)
    keepdim_shape = tuple(1 if index == dim else axis for index, axis in enumerate(input_shape))
    if not (_shape_equal(output_shape, reduced_shape) or _shape_equal(output_shape, keepdim_shape)):
        raise ValueError(f"{op_name} output shape {output_shape} is not a valid reduction of {input_shape} on dim {dim}.")


def _ensure_shape_pair_compatible(
    lhs_shape: tuple[Any, ...] | None,
    rhs_shape: tuple[Any, ...] | None,
    description: str,
) -> None:
    if lhs_shape is None or rhs_shape is None:
        return
    if not (_shape_equal(lhs_shape, rhs_shape) or _volume_compatible(lhs_shape, rhs_shape)):
        raise ValueError(f"{description} shape mismatch: {lhs_shape} vs {rhs_shape}.")


def _can_broadcast(input_shape: tuple[Any, ...], output_shape: tuple[Any, ...]) -> bool:
    if _shape_equal(input_shape, output_shape):
        return True
    if len(input_shape) == 1 and len(output_shape) > 1 and _dim_equal(input_shape[0], output_shape[0]):
        return True
    reversed_input = list(reversed(input_shape))
    reversed_output = list(reversed(output_shape))
    for index, output_dim in enumerate(reversed_output):
        input_dim = reversed_input[index] if index < len(reversed_input) else 1
        if not (_dim_equal(input_dim, 1) or _dim_equal(input_dim, output_dim)):
            return False
    return True


def _shape_equal(lhs_shape: tuple[Any, ...], rhs_shape: tuple[Any, ...]) -> bool:
    return len(lhs_shape) == len(rhs_shape) and all(_dim_equal(lhs, rhs) for lhs, rhs in zip(lhs_shape, rhs_shape))


def _volume_compatible(lhs_shape: tuple[Any, ...], rhs_shape: tuple[Any, ...]) -> bool:
    lhs_volume = _shape_volume(lhs_shape)
    rhs_volume = _shape_volume(rhs_shape)
    return _dim_equal(lhs_volume, rhs_volume)


def _shape_volume(shape: tuple[Any, ...]) -> Any:
    volume: Any = 1
    for extent in shape:
        if extent is None:
            return None
        volume = volume * extent
    return volume


def _dim_equal(lhs: Any, rhs: Any) -> bool:
    if lhs is None or rhs is None:
        return True
    if isinstance(lhs, int) and isinstance(rhs, int):
        return lhs == rhs
    comparison = lhs == rhs
    return comparison if isinstance(comparison, bool) else True


# Shape helpers

def _base_strides(base: Any) -> tuple[Any, ...] | None:
    strides = getattr(base, "strides", None)
    if strides is not None:
        return tuple(strides)
    shape = getattr(base, "shape", None)
    if shape is None:
        return None
    return _contiguous_strides(tuple(shape))


def _contiguous_strides(shape: tuple[Any, ...]) -> tuple[Any, ...]:
    stride = 1
    strides: list[Any] = []
    for extent in reversed(shape):
        strides.insert(0, stride)
        stride = stride * extent
    return tuple(strides)


def _point_region_strides(base_strides: tuple[Any, ...] | None, rank: int) -> tuple[Any, ...]:
    if base_strides is not None and len(base_strides) >= rank:
        return tuple(base_strides[:rank])
    return _contiguous_strides(tuple(1 for _ in range(rank)))


def _slice_extent(start: Any, stop: Any, step: Any) -> Any:
    if stop is None:
        return None
    if step == 1:
        return stop - start
    if all(isinstance(value, int) for value in (start, stop, step)):
        return math.ceil((stop - start) / step)
    return (stop - start + step - 1) // step


def _byte_count(shape: tuple[Any, ...] | None, dtype: Any) -> Any | None:
    if shape is None or dtype is None or not all(isinstance(extent, int) for extent in shape):
        return None
    element_count = 1
    for extent in shape:
        element_count *= extent
    return element_count * dtype_byte_size(dtype)


def _memory_space(base: Any) -> str:
    if isinstance(base, Buffer):
        return "local"
    return "tensor"


def _is_buffer_like(value: Any) -> bool:
    return isinstance(value, (Tensor, TensorRegion, Buffer, BufferRegion))

