"""Shared helpers for the public atlang language surface."""

from __future__ import annotations

from typing import Any


def normalize_shape(shape: Any) -> tuple[Any, ...]:
    if isinstance(shape, tuple):
        return shape
    if isinstance(shape, list):
        return tuple(shape)
    return (shape,)


def normalize_indices(key: Any) -> tuple[Any, ...]:
    if isinstance(key, tuple):
        return key
    return (key,)


def reject_kwargs(name: str, kwargs: dict[str, Any]) -> None:
    if not kwargs:
        return
    joined = ", ".join(sorted(kwargs))
    raise TypeError(f"{name} got unsupported atlang kwargs: {joined}.")


__all__ = ["normalize_indices", "normalize_shape", "reject_kwargs"]
