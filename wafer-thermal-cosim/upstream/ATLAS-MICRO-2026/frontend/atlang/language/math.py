"""Scalar math helpers that are part of the public atlang language surface."""

from __future__ import annotations

import math as _math
from typing import Any

from ..ir.dtype import dtype_of
from ..ir.expr import ceildiv as _expr_ceildiv
from .utils import reject_kwargs


# Scalar helpers

def ceildiv(lhs: Any, rhs: Any, **kwargs: Any) -> Any:
    reject_kwargs("ceildiv", kwargs)
    return _expr_ceildiv(lhs, rhs)


def infinity(dtype: Any | None = None) -> float:
    if dtype is not None:
        dtype_of(dtype)
    return _math.inf


__all__ = ["ceildiv", "infinity"]
