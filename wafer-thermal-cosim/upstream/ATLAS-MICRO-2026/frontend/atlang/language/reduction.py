"""Reduction helper functions for atlang source kernels."""

from __future__ import annotations

import builtins
from typing import Any, Callable

from ..ir.expr import CallExpr
from .objects import ProcessStyleCall
from .utils import reject_kwargs


# Reduction helpers

def reduce(
    buffer: Any,
    out: Any = None,
    reduce_type: str = "sum",
    dim: int = 0,
    clear: builtins.bool = True,
    **kwargs: Any,
) -> Any:
    reject_kwargs("reduce", kwargs)
    if out is None:
        return CallExpr("reduce", (buffer,))
    return ProcessStyleCall(
        op="reduce",
        args=(buffer,),
        out=out,
        attrs={"reduce_type": reduce_type, "dim": dim, "clear": clear},
    )


def cumsum(
    buffer: Any,
    out: Any = None,
    dim: int = 0,
    reverse: builtins.bool = False,
    **kwargs: Any,
) -> Any:
    reject_kwargs("cumsum", kwargs)
    if out is None:
        return CallExpr("cumsum", (buffer,))
    return ProcessStyleCall(op="cumsum", args=(buffer,), out=out, attrs={"dim": dim, "reverse": reverse})


def _make_reduce_function(reduce_type: str) -> Callable[..., Any]:
    def operation(buffer: Any, out: Any = None, dim: int = 0, clear: builtins.bool = True, **kwargs: Any) -> Any:
        return reduce(buffer, out=out, reduce_type=reduce_type, dim=dim, clear=clear, **kwargs)

    operation.__name__ = f"reduce_{reduce_type}"
    return operation

reduce_max = _make_reduce_function("max")
reduce_min = _make_reduce_function("min")
reduce_sum = _make_reduce_function("sum")
reduce_abssum = _make_reduce_function("abssum")
reduce_absmax = _make_reduce_function("absmax")
reduce_bitand = _make_reduce_function("bitand")
reduce_bitor = _make_reduce_function("bitor")
reduce_bitxor = _make_reduce_function("bitxor")


__all__ = [
    "cumsum",
    "reduce",
    "reduce_absmax",
    "reduce_abssum",
    "reduce_bitand",
    "reduce_bitor",
    "reduce_bitxor",
    "reduce_max",
    "reduce_min",
    "reduce_sum",
]
