"""Scalar and vector helper functions for atlang source kernels."""

from __future__ import annotations

import builtins
import math
import operator
from typing import Any, Callable

from ..ir.dtype import DType, dtype_of
from ..ir.expr import CallExpr, is_expr, make_binary_expr, make_call_expr
from .objects import Buffer, BufferRegion, ProcessStyleCall, Tensor, TensorRegion
from .utils import reject_kwargs


# Scalar and vector helpers

_BINARY_EXPR_OPS = {
    "add": "add",
    "sub": "sub",
    "mul": "mul",
    "div": "truediv",
    "floordiv": "floordiv",
    "floormod": "mod",
}

_BINARY_STATIC_OPS: dict[str, Callable[..., Any]] = {
    "add": operator.add,
    "sub": operator.sub,
    "mul": operator.mul,
    "div": operator.truediv,
    "max": builtins.max,
    "min": builtins.min,
    "atan2": math.atan2,
    "bitwise_and": operator.and_,
    "bitwise_or": operator.or_,
    "bitwise_xor": operator.xor,
    "copysign": math.copysign,
    "floordiv": operator.floordiv,
    "floormod": operator.mod,
    "fmod": math.fmod,
    "hypot": math.hypot,
    "ldexp": math.ldexp,
    "nextafter": math.nextafter,
    "pow": builtins.pow,
    "shift_left": operator.lshift,
    "shift_right": operator.rshift,
}

_UNARY_STATIC_OPS: dict[str, Callable[[Any], Any]] = {
    "abs": builtins.abs,
    "acos": math.acos,
    "acosh": math.acosh,
    "asin": math.asin,
    "asinh": math.asinh,
    "atan": math.atan,
    "atanh": math.atanh,
    "bitwise_not": operator.invert,
    "ceil": math.ceil,
    "cos": math.cos,
    "cosh": math.cosh,
    "erf": math.erf,
    "exp": math.exp,
    "exp2": lambda value: 2 ** value,
    "exp10": lambda value: 10 ** value,
    "floor": math.floor,
    "isfinite": math.isfinite,
    "isinf": math.isinf,
    "isnan": math.isnan,
    "isnullptr": lambda value: value is None,
    "likely": lambda value: value,
    "log": math.log,
    "log1p": math.log1p,
    "log2": math.log2,
    "log10": math.log10,
    "nearbyint": builtins.round,
    "popcount": lambda value: int(value).bit_count(),
    "round": builtins.round,
    "rsqrt": lambda value: 1 / math.sqrt(value),
    "sigmoid": lambda value: 1 / (1 + math.exp(-value)),
    "sin": math.sin,
    "sinh": math.sinh,
    "sqrt": math.sqrt,
    "tan": math.tan,
    "tanh": math.tanh,
    "trunc": math.trunc,
}


def cast(value: Any, dtype_or_out: Any = None, out: Any = None, **kwargs: Any) -> Any:
    reject_kwargs("cast", kwargs)
    if out is not None:
        return ProcessStyleCall(op="cast", args=(value,), out=out, attrs={"dtype": _optional_dtype(dtype_or_out)})
    if _is_buffer_like(dtype_or_out):
        return ProcessStyleCall(op="cast", args=(value,), out=dtype_or_out, attrs={})
    if is_expr(value):
        return CallExpr("cast", (value, dtype_or_out))
    return value


def reinterpret(value: Any, dtype_or_out: Any = None, out: Any = None, **kwargs: Any) -> Any:
    reject_kwargs("reinterpret", kwargs)
    if out is not None:
        return ProcessStyleCall(op="reinterpret", args=(value,), out=out, attrs={"dtype": _optional_dtype(dtype_or_out)})
    if _is_buffer_like(dtype_or_out):
        return ProcessStyleCall(op="reinterpret", args=(value,), out=dtype_or_out, attrs={})
    if is_expr(value):
        return CallExpr("reinterpret", (value, dtype_or_out))
    return value


def if_then_else(condition: Any, true_value: Any, false_value: Any, *extra_args: Any, **kwargs: Any) -> Any:
    reject_kwargs("if_then_else", kwargs)
    args = (condition, true_value, false_value, *extra_args)
    if _is_process_style_args(args):
        return ProcessStyleCall(op="if_then_else", args=args[:-1], out=args[-1])
    if any(is_expr(arg) for arg in args):
        return make_call_expr("if_then_else", condition, true_value, false_value)
    return true_value if builtins.bool(condition) else false_value


def truncdiv(lhs: Any, rhs: Any, *extra_args: Any, **kwargs: Any) -> Any:
    reject_kwargs("truncdiv", kwargs)
    args = (lhs, rhs, *extra_args)
    if _is_process_style_args(args):
        return ProcessStyleCall(op="truncdiv", args=args[:-1], out=args[-1])
    if any(is_expr(arg) for arg in args):
        return CallExpr("truncdiv", args)
    return math.trunc(lhs / rhs)


def truncmod(lhs: Any, rhs: Any, *extra_args: Any, **kwargs: Any) -> Any:
    reject_kwargs("truncmod", kwargs)
    args = (lhs, rhs, *extra_args)
    if _is_process_style_args(args):
        return ProcessStyleCall(op="truncmod", args=args[:-1], out=args[-1])
    if any(is_expr(arg) for arg in args):
        return CallExpr("truncmod", args)
    return lhs - math.trunc(lhs / rhs) * rhs


def q_multiply_shift(*args: Any, **kwargs: Any) -> Any:
    return _nary_process_or_scalar("q_multiply_shift", args, kwargs)


def q_multiply_shift_per_axis(*args: Any, **kwargs: Any) -> Any:
    return _nary_process_or_scalar("q_multiply_shift_per_axis", args, kwargs)


def _make_binary_function(op: str) -> Callable[..., Any]:
    def operation(lhs: Any, rhs: Any, out: Any = None, **kwargs: Any) -> Any:
        reject_kwargs(op, kwargs)
        if out is not None:
            return ProcessStyleCall(op=op, args=(lhs, rhs), out=out)
        return _binary_scalar(op, lhs, rhs)

    operation.__name__ = op
    return operation


def _make_unary_function(op: str) -> Callable[..., Any]:
    def operation(value: Any, out: Any = None, **kwargs: Any) -> Any:
        reject_kwargs(op, kwargs)
        if out is not None:
            return ProcessStyleCall(op=op, args=(value,), out=out)
        if is_expr(value):
            return CallExpr(op, (value,))
        evaluator = _UNARY_STATIC_OPS.get(op)
        if evaluator is None:
            return CallExpr(op, (value,))
        return evaluator(value)

    operation.__name__ = op
    return operation


def _make_nary_function(op: str) -> Callable[..., Any]:
    def operation(*args: Any, **kwargs: Any) -> Any:
        return _nary_process_or_scalar(op, args, kwargs)

    operation.__name__ = op
    return operation


def _binary_scalar(op: str, lhs: Any, rhs: Any) -> Any:
    expression_op = _BINARY_EXPR_OPS.get(op)
    if expression_op is not None:
        return make_binary_expr(expression_op, lhs, rhs)
    if is_expr(lhs) or is_expr(rhs):
        return CallExpr(op, (lhs, rhs))
    evaluator = _BINARY_STATIC_OPS.get(op)
    if evaluator is None:
        return CallExpr(op, (lhs, rhs))
    return evaluator(lhs, rhs)


def _nary_process_or_scalar(op: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    reject_kwargs(op, kwargs)
    if _is_process_style_args(args):
        return ProcessStyleCall(op=op, args=args[:-1], out=args[-1])
    if len(args) == 2:
        return _binary_scalar(op, args[0], args[1])
    if any(is_expr(arg) for arg in args):
        return CallExpr(op, args)
    evaluator = _BINARY_STATIC_OPS.get(op)
    if evaluator is not None:
        return evaluator(*args)
    return CallExpr(op, args)


def _is_process_style_args(args: tuple[Any, ...]) -> builtins.bool:
    return len(args) >= 2 and _is_buffer_like(args[-1]) and any(_is_buffer_like(arg) for arg in args[:-1])


def _optional_dtype(value: Any) -> DType | None:
    if value is None or _is_buffer_like(value):
        return None
    return dtype_of(value)


def _is_buffer_like(value: Any) -> builtins.bool:
    return isinstance(value, (Tensor, TensorRegion, Buffer, BufferRegion))


for _binary_name in ("add", "sub", "mul", "div", "max", "min"):
    globals()[_binary_name] = _make_binary_function(_binary_name)

for _unary_name in (
    "abs",
    "acos",
    "acosh",
    "asin",
    "asinh",
    "atan",
    "atanh",
    "bitwise_not",
    "ceil",
    "clz",
    "cos",
    "cosh",
    "erf",
    "exp",
    "exp2",
    "exp10",
    "floor",
    "isfinite",
    "isinf",
    "isnan",
    "isnullptr",
    "likely",
    "log",
    "log1p",
    "log2",
    "log10",
    "nearbyint",
    "popcount",
    "round",
    "rsqrt",
    "sigmoid",
    "sin",
    "sinh",
    "sqrt",
    "tan",
    "tanh",
    "trunc",
):
    globals()[_unary_name] = _make_unary_function(_unary_name)

for _nary_name in (
    "atan2",
    "bitwise_and",
    "bitwise_or",
    "bitwise_xor",
    "copysign",
    "floordiv",
    "floormod",
    "fmod",
    "hypot",
    "ldexp",
    "nextafter",
    "pow",
    "shift_left",
    "shift_right",
):
    globals()[_nary_name] = _make_nary_function(_nary_name)


__all__ = [
    "abs",
    "acos",
    "acosh",
    "add",
    "asin",
    "asinh",
    "atan",
    "atan2",
    "atanh",
    "bitwise_and",
    "bitwise_not",
    "bitwise_or",
    "bitwise_xor",
    "cast",
    "ceil",
    "clz",
    "copysign",
    "cos",
    "cosh",
    "div",
    "erf",
    "exp",
    "exp10",
    "exp2",
    "floor",
    "floordiv",
    "floormod",
    "fmod",
    "hypot",
    "if_then_else",
    "isfinite",
    "isinf",
    "isnan",
    "isnullptr",
    "ldexp",
    "likely",
    "log",
    "log10",
    "log1p",
    "log2",
    "max",
    "min",
    "mul",
    "nearbyint",
    "nextafter",
    "popcount",
    "pow",
    "q_multiply_shift",
    "q_multiply_shift_per_axis",
    "reinterpret",
    "round",
    "rsqrt",
    "shift_left",
    "shift_right",
    "sigmoid",
    "sin",
    "sinh",
    "sqrt",
    "sub",
    "tan",
    "tanh",
    "trunc",
    "truncdiv",
    "truncmod",
]
