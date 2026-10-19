"""Static expression helpers shared by atlang extraction materializers."""

from __future__ import annotations

from typing import Any

from ...ir.expr import (
    collect_ceildiv_operands,
    evaluate_static_bool as _evaluate_static_bool,
    evaluate_static_int as _evaluate_static_int,
    replace_ceildiv_with_static_extents,
    substitute_expr,
)


# Static substitution and evaluation

def substitute_and_simplify(expr: Any, env: dict[str, Any]) -> Any:
    return substitute_expr(expr, env)


def evaluate_static_int(expr: Any, env: dict[str, Any], *, description: str) -> int:
    return _evaluate_static_int(expr, env, description=description)


def evaluate_static_bool(expr: Any, env: dict[str, Any], *, description: str) -> bool:
    return _evaluate_static_bool(expr, env, description=description)


__all__ = [
    "collect_ceildiv_operands",
    "evaluate_static_bool",
    "evaluate_static_int",
    "replace_ceildiv_with_static_extents",
    "substitute_and_simplify",
]
