"""GEMM operation call in the atlang language surface."""

from __future__ import annotations

import builtins
from typing import Any

from .objects import ProcessStyleCall
from .utils import reject_kwargs


# GEMM operation

def gemm(
    A: Any,
    B: Any,
    C: Any,
    transpose_A: builtins.bool = False,
    transpose_B: builtins.bool = False,
    **kwargs: Any,
) -> ProcessStyleCall:
    reject_kwargs("gemm", kwargs)
    return ProcessStyleCall(
        op="gemm",
        args=(A, B),
        out=C,
        attrs={"transpose_A": transpose_A, "transpose_B": transpose_B},
    )


__all__ = ["gemm"]
