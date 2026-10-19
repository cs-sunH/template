"""Scalar normalization helpers for atlang extraction."""

from __future__ import annotations

from typing import Any

from ...ir.expr import is_expr
from .expressions import evaluate_static_int


def unwrap_int_value(value: Any, env: dict[str, Any] | None = None, *, description: str = "value") -> int:
    if isinstance(value, bool):
        raise ValueError(f"{description} must be an integer, got bool.")
    if isinstance(value, int):
        return int(value)
    if is_expr(value):
        return evaluate_static_int(value, env or {}, description=description)
    if hasattr(value, "value") and isinstance(value.value, int):
        return int(value.value)
    raise ValueError(f"{description} must be an integer, got {value!r}.")


def to_python_int(value: Any, *, description: str = "value") -> int:
    return unwrap_int_value(value, description=description)


def positive_int_from_core_array_kwargs(core_array_kwargs: dict[str, Any], key: str) -> int:
    value = core_array_kwargs.get(key)
    if value is None:
        return 1
    try:
        parsed = to_python_int(value, description=f"core_array_kwargs[{key!r}]")
    except ValueError:
        return 1
    return parsed if parsed > 0 else 1


__all__ = ["positive_int_from_core_array_kwargs", "to_python_int", "unwrap_int_value"]
