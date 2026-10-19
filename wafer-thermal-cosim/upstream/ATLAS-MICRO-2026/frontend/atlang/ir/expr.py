"""Symbolic expression helpers for atlang capture and extraction."""

from __future__ import annotations

import math
import operator
from dataclasses import dataclass
from typing import Any, Iterable


# Expression nodes

class Expr:
    """Base class for neutral symbolic expressions."""

    def __bool__(self) -> bool:
        raise TypeError("Atlang symbolic expressions cannot be used as Python booleans directly.")

    def __add__(self, other: Any) -> Any:
        return make_binary_expr("add", self, other)

    def __radd__(self, other: Any) -> Any:
        return make_binary_expr("add", other, self)

    def __sub__(self, other: Any) -> Any:
        return make_binary_expr("sub", self, other)

    def __rsub__(self, other: Any) -> Any:
        return make_binary_expr("sub", other, self)

    def __mul__(self, other: Any) -> Any:
        return make_binary_expr("mul", self, other)

    def __rmul__(self, other: Any) -> Any:
        return make_binary_expr("mul", other, self)

    def __truediv__(self, other: Any) -> Any:
        return make_binary_expr("truediv", self, other)

    def __rtruediv__(self, other: Any) -> Any:
        return make_binary_expr("truediv", other, self)

    def __floordiv__(self, other: Any) -> Any:
        return make_binary_expr("floordiv", self, other)

    def __rfloordiv__(self, other: Any) -> Any:
        return make_binary_expr("floordiv", other, self)

    def __mod__(self, other: Any) -> Any:
        return make_binary_expr("mod", self, other)

    def __rmod__(self, other: Any) -> Any:
        return make_binary_expr("mod", other, self)

    def __neg__(self) -> Any:
        return make_unary_expr("neg", self)

    def __pos__(self) -> Any:
        return make_unary_expr("pos", self)

    def __lt__(self, other: Any) -> Any:
        return make_binary_expr("lt", self, other)

    def __le__(self, other: Any) -> Any:
        return make_binary_expr("le", self, other)

    def __gt__(self, other: Any) -> Any:
        return make_binary_expr("gt", self, other)

    def __ge__(self, other: Any) -> Any:
        return make_binary_expr("ge", self, other)

    def __eq__(self, other: Any) -> Any:  # type: ignore[override]
        return make_binary_expr("eq", self, other)

    def __ne__(self, other: Any) -> Any:  # type: ignore[override]
        return make_binary_expr("ne", self, other)

    def __getitem__(self, key: Any) -> Any:
        return deferred_getitem(self, key)


@dataclass(frozen=True, slots=True, eq=False)
class LiteralExpr(Expr):
    value: Any

    def __repr__(self) -> str:
        return repr(self.value)


@dataclass(frozen=True, slots=True, eq=False)
class SymbolExpr(Expr):
    name: str

    def __repr__(self) -> str:
        return self.name


@dataclass(frozen=True, slots=True, eq=False)
class UnaryExpr(Expr):
    op: str
    value: Any

    def __repr__(self) -> str:
        return f"{self.op}({self.value!r})"


@dataclass(frozen=True, slots=True, eq=False)
class BinaryExpr(Expr):
    op: str
    lhs: Any
    rhs: Any

    def __repr__(self) -> str:
        return f"({self.lhs!r} {self.op} {self.rhs!r})"


@dataclass(frozen=True, slots=True, eq=False)
class CallExpr(Expr):
    op: str
    args: tuple[Any, ...]

    def __repr__(self) -> str:
        args = ", ".join(repr(arg) for arg in self.args)
        return f"{self.op}({args})"


@dataclass(frozen=True, slots=True, eq=False)
class CeilDivExpr(Expr):
    lhs: Any
    rhs: Any

    def __repr__(self) -> str:
        return f"ceildiv({self.lhs!r}, {self.rhs!r})"


@dataclass(frozen=True, slots=True, eq=False)
class DeferredIndexExpr(Expr):
    container: Any
    key: Any

    def __repr__(self) -> str:
        return f"{self.container!r}[{self.key!r}]"


# Construction helpers

def is_expr(value: Any) -> bool:
    return isinstance(value, Expr)


def literal(value: Any) -> Any:
    if is_expr(value):
        return value
    return LiteralExpr(value)


def symbol(name: str) -> SymbolExpr:
    return SymbolExpr(str(name))


def _has_expr(values: Iterable[Any]) -> bool:
    return any(is_expr(value) for value in values)


def make_unary_expr(op: str, value: Any) -> Any:
    if not is_expr(value):
        return _eval_unary(op, value)
    return UnaryExpr(op, value)


def make_binary_expr(op: str, lhs: Any, rhs: Any) -> Any:
    if not _has_expr((lhs, rhs)):
        return _eval_binary(op, lhs, rhs)
    return BinaryExpr(op, lhs, rhs)


def make_call_expr(op: str, *args: Any) -> Any:
    if not _has_expr(args):
        return _eval_call(op, args)
    return CallExpr(op, tuple(args))


def ceildiv(lhs: Any, rhs: Any) -> CeilDivExpr:
    return CeilDivExpr(lhs, rhs)


def deferred_getitem(container: Any, key: Any) -> Any:
    if not is_expr(container) and not is_expr(key):
        return container[key]
    return DeferredIndexExpr(container, key)


def static_len(value: Any) -> int:
    if is_expr(value):
        raise ValueError(f"len() requires a concrete container, got {value!r}.")
    return len(value)


# Static evaluation

_BINARY_OPERATORS = {
    "add": operator.add,
    "sub": operator.sub,
    "mul": operator.mul,
    "truediv": operator.truediv,
    "floordiv": operator.floordiv,
    "mod": operator.mod,
    "lt": operator.lt,
    "le": operator.le,
    "gt": operator.gt,
    "ge": operator.ge,
    "eq": operator.eq,
    "ne": operator.ne,
    "and": lambda lhs, rhs: bool(lhs) and bool(rhs),
    "or": lambda lhs, rhs: bool(lhs) or bool(rhs),
}

_UNARY_OPERATORS = {
    "neg": operator.neg,
    "pos": operator.pos,
    "not": operator.not_,
}

_CALL_OPERATORS = {
    "abs": abs,
    "len": len,
    "max": max,
    "min": min,
}


def _eval_unary(op: str, value: Any) -> Any:
    try:
        return _UNARY_OPERATORS[op](value)
    except KeyError as exc:
        raise ValueError(f"Unsupported unary expression op {op!r}.") from exc


def _eval_binary(op: str, lhs: Any, rhs: Any) -> Any:
    try:
        return _BINARY_OPERATORS[op](lhs, rhs)
    except KeyError as exc:
        raise ValueError(f"Unsupported binary expression op {op!r}.") from exc


def _eval_call(op: str, args: tuple[Any, ...]) -> Any:
    if op == "if_then_else":
        condition, true_value, false_value = args
        return true_value if bool(condition) else false_value
    try:
        return _CALL_OPERATORS[op](*args)
    except KeyError as exc:
        raise ValueError(f"Unsupported call expression op {op!r}.") from exc


def substitute_expr(value: Any, env: dict[str, Any]) -> Any:
    if isinstance(value, LiteralExpr):
        return value.value
    if isinstance(value, SymbolExpr):
        return env.get(value.name, value)
    if isinstance(value, UnaryExpr):
        substituted = substitute_expr(value.value, env)
        return make_unary_expr(value.op, substituted)
    if isinstance(value, BinaryExpr):
        lhs = substitute_expr(value.lhs, env)
        rhs = substitute_expr(value.rhs, env)
        return make_binary_expr(value.op, lhs, rhs)
    if isinstance(value, CallExpr):
        args = tuple(substitute_expr(arg, env) for arg in value.args)
        return make_call_expr(value.op, *args)
    if isinstance(value, CeilDivExpr):
        lhs = substitute_expr(value.lhs, env)
        rhs = substitute_expr(value.rhs, env)
        if not _has_expr((lhs, rhs)):
            return _ceildiv_value(lhs, rhs)
        return CeilDivExpr(lhs, rhs)
    if isinstance(value, DeferredIndexExpr):
        key = substitute_expr(value.key, env)
        container = substitute_container(value.container, env)
        if not is_expr(container) and not is_expr(key):
            return container[key]
        return DeferredIndexExpr(container, key)
    return value


def substitute_container(value: Any, env: dict[str, Any]) -> Any:
    if isinstance(value, list):
        return [substitute_container(item, env) for item in value]
    if isinstance(value, tuple):
        return tuple(substitute_container(item, env) for item in value)
    if isinstance(value, dict):
        return {substitute_container(key, env): substitute_container(item, env) for key, item in value.items()}
    return substitute_expr(value, env)


def evaluate_static_int(value: Any, env: dict[str, Any] | None = None, *, description: str = "value") -> int:
    resolved = substitute_expr(value, env or {})
    if isinstance(resolved, bool):
        raise ValueError(f"{description} must resolve to a static integer, but got bool.")
    if isinstance(resolved, int):
        return int(resolved)
    raise ValueError(f"{description} must resolve to a static integer, but got {resolved!r}.")


def evaluate_static_bool(value: Any, env: dict[str, Any] | None = None, *, description: str = "predicate") -> bool:
    resolved = substitute_expr(value, env or {})
    if isinstance(resolved, bool):
        return resolved
    if isinstance(resolved, int):
        return bool(resolved)
    raise ValueError(f"{description} must resolve to a static predicate, but got {resolved!r}.")


# Traversal and autotune helpers

def walk_expr(value: Any) -> Iterable[Any]:
    yield value
    if isinstance(value, UnaryExpr):
        yield from walk_expr(value.value)
    elif isinstance(value, BinaryExpr):
        yield from walk_expr(value.lhs)
        yield from walk_expr(value.rhs)
    elif isinstance(value, CallExpr):
        for arg in value.args:
            yield from walk_expr(arg)
    elif isinstance(value, CeilDivExpr):
        yield from walk_expr(value.lhs)
        yield from walk_expr(value.rhs)
    elif isinstance(value, DeferredIndexExpr):
        yield from walk_expr(value.container)
        yield from walk_expr(value.key)


def collect_symbols(value: Any) -> set[str]:
    return {node.name for node in walk_expr(value) if isinstance(node, SymbolExpr)}


def collect_ceildiv_operands(value: Any) -> list[tuple[Any, Any]]:
    return [(node.lhs, node.rhs) for node in walk_expr(value) if isinstance(node, CeilDivExpr)]


def replace_ceildiv_with_static_extents(value: Any, rhs_values: list[int], env: dict[str, Any] | None = None) -> Any:
    env = env or {}
    next_index = 0

    def replace(current: Any) -> Any:
        nonlocal next_index
        if isinstance(current, CeilDivExpr):
            if next_index >= len(rhs_values):
                raise ValueError("Ceildiv replacement did not provide enough rhs values.")
            lhs_value = evaluate_static_int(current.lhs, env, description="ceildiv lhs")
            rhs_value = int(rhs_values[next_index])
            next_index += 1
            if rhs_value <= 0 or lhs_value % rhs_value != 0:
                raise ValueError(f"Replacement rhs {rhs_value} must be a positive divisor of lhs {lhs_value}.")
            return lhs_value // rhs_value
        if isinstance(current, UnaryExpr):
            return make_unary_expr(current.op, replace(current.value))
        if isinstance(current, BinaryExpr):
            return make_binary_expr(current.op, replace(current.lhs), replace(current.rhs))
        if isinstance(current, CallExpr):
            return make_call_expr(current.op, *(replace(arg) for arg in current.args))
        if isinstance(current, DeferredIndexExpr):
            return deferred_getitem(current.container, replace(current.key))
        return substitute_expr(current, env)

    replaced = replace(value)
    if next_index != len(rhs_values):
        raise ValueError("Ceildiv replacement provided unused rhs values.")
    return replaced


def _ceildiv_value(lhs: Any, rhs: Any) -> int:
    lhs_value = int(lhs)
    rhs_value = int(rhs)
    if rhs_value == 0:
        raise ZeroDivisionError("ceildiv divisor cannot be zero.")
    return int(math.ceil(lhs_value / rhs_value))


__all__ = [
    "BinaryExpr",
    "CallExpr",
    "CeilDivExpr",
    "DeferredIndexExpr",
    "Expr",
    "LiteralExpr",
    "SymbolExpr",
    "UnaryExpr",
    "ceildiv",
    "collect_ceildiv_operands",
    "collect_symbols",
    "deferred_getitem",
    "evaluate_static_bool",
    "evaluate_static_int",
    "is_expr",
    "literal",
    "make_binary_expr",
    "make_call_expr",
    "make_unary_expr",
    "replace_ceildiv_with_static_extents",
    "static_len",
    "substitute_expr",
    "symbol",
    "walk_expr",
]
