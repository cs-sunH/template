"""Memory movement and initialization calls in the atlang language surface."""

from __future__ import annotations

from typing import Any

from ..ir.dtype import dtype_of, float32
from .objects import Buffer, ProcessStyleCall
from .utils import normalize_shape, reject_kwargs


# Memory movement and initialization

def alloc(shape: Any, dtype: Any = float32, **kwargs: Any) -> Buffer:
    reject_kwargs("alloc", kwargs)
    return Buffer(shape=normalize_shape(shape), dtype=dtype_of(dtype), name=None, attrs={})


def copy(src: Any, dst: Any, **kwargs: Any) -> ProcessStyleCall:
    reject_kwargs("copy", kwargs)
    return ProcessStyleCall(op="copy", args=(src,), out=dst)


def fill(buffer: Any, value: Any, **kwargs: Any) -> ProcessStyleCall:
    reject_kwargs("fill", kwargs)
    return ProcessStyleCall(op="fill", args=(value,), out=buffer)


def clear(buffer: Any, **kwargs: Any) -> ProcessStyleCall:
    reject_kwargs("clear", kwargs)
    return ProcessStyleCall(op="clear", out=buffer)


__all__ = ["alloc", "clear", "copy", "fill"]
