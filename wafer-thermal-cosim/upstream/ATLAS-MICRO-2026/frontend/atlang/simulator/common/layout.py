"""Layout helpers shared by atlang extraction modules."""

from __future__ import annotations

import math
from typing import Any

from ...ir.nodes import TensorDecl
from .scalars import unwrap_int_value


# Layout normalization

def normalize_layout_tuple(values: tuple[Any, ...] | None, *, field_name: str, tensor_name: str) -> tuple[int, ...]:
    if values is None:
        raise ValueError(f"Tensor '{tensor_name}' is missing {field_name}.")
    normalized = tuple(unwrap_int_value(value, description=f"{tensor_name}.{field_name}") for value in values)
    if not all(isinstance(value, int) for value in normalized):
        raise ValueError(f"Tensor '{tensor_name}' has non-integer {field_name}: {normalized}")
    return normalized


def require_2d_decl(tensor_decl: TensorDecl) -> tuple[tuple[int, int], tuple[int, int]]:
    shape = normalize_layout_tuple(tensor_decl.shape, field_name="shape", tensor_name=tensor_decl.name)
    strides = normalize_layout_tuple(tensor_decl.strides, field_name="strides", tensor_name=tensor_decl.name)
    if len(shape) != 2 or len(strides) != 2:
        raise ValueError(
            f"Tensor '{tensor_decl.name}' must be 2D for this extraction stage, got shape={shape}, strides={strides}"
        )
    return (shape[0], shape[1]), (strides[0], strides[1])


def tensor_volume_bytes(shape: tuple[int, ...] | list[int], element_size: int) -> int:
    volume = 1
    for dim in shape:
        volume *= int(dim)
    return volume * int(element_size)


def align_base_addr(base_addr: int, volume_bytes: int, dram_row_size: int) -> int:
    return int(base_addr) + math.ceil(int(volume_bytes) / int(dram_row_size)) * int(dram_row_size)


def build_layout_record(
    name: str,
    base_addr: int,
    shape: tuple[int, ...] | list[int],
    strides: tuple[int, ...] | list[int],
    element_size: int,
) -> dict[str, Any]:
    return {
        "name": name,
        "base_addr": int(base_addr),
        "shape": [int(dim) for dim in shape],
        "strides": [int(dim) for dim in strides],
        "element_size": int(element_size),
    }


__all__ = ["align_base_addr", "build_layout_record", "normalize_layout_tuple", "require_2d_decl", "tensor_volume_bytes"]
