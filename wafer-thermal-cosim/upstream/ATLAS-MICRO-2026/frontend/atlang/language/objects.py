"""Public placeholder objects used by atlang source kernels."""

from __future__ import annotations

import builtins
from dataclasses import dataclass, field
from typing import Any, Iterable

from ..ir.dtype import DType, dtype_of, float32
from ..ir.expr import is_expr, symbol
from .utils import normalize_indices, normalize_shape, reject_kwargs


# Value objects

@dataclass(frozen=True, slots=True)
class Tensor:
    """Tensor layout declaration used by source kernels."""

    shape: tuple[Any, ...]
    dtype: DType = float32
    data: Any = None
    strides: tuple[Any, ...] | None = None
    name: str | None = None
    attrs: dict[str, Any] = field(default_factory=dict)

    def __init__(
        self,
        shape: Any,
        dtype: Any = float32,
        data: Any = None,
        strides: Any = None,
        name: str | None = None,
        **kwargs: Any,
    ) -> None:
        reject_kwargs("Tensor", kwargs)
        object.__setattr__(self, "shape", normalize_shape(shape))
        object.__setattr__(self, "dtype", dtype_of(dtype))
        object.__setattr__(self, "data", data)
        object.__setattr__(self, "strides", None if strides is None else normalize_shape(strides))
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "attrs", {})

    def __getitem__(self, key: Any) -> "TensorRegion":
        return TensorRegion(base=self, indices=normalize_indices(key))


@dataclass(frozen=True, slots=True)
class TensorRegion:
    """Indexed tensor region placeholder."""

    base: Any
    indices: tuple[Any, ...]

    @property
    def name(self) -> str | None:
        return getattr(self.base, "name", None)

    @property
    def shape(self) -> tuple[Any, ...]:
        return getattr(self.base, "shape", ())

    @property
    def dtype(self) -> DType | None:
        return getattr(self.base, "dtype", None)

    def __getitem__(self, key: Any) -> "TensorRegion":
        return TensorRegion(base=self, indices=normalize_indices(key))


@dataclass(frozen=True, slots=True)
class Buffer:
    """Temporary local buffer placeholder returned by alloc."""

    shape: tuple[Any, ...]
    dtype: DType = float32
    name: str | None = None
    attrs: dict[str, Any] = field(default_factory=dict)

    def __getitem__(self, key: Any) -> "BufferRegion":
        return BufferRegion(base=self, indices=normalize_indices(key))


@dataclass(frozen=True, slots=True)
class BufferRegion:
    """Indexed temporary buffer region placeholder."""

    base: Any
    indices: tuple[Any, ...]

    @property
    def name(self) -> str | None:
        return getattr(self.base, "name", None)

    @property
    def shape(self) -> tuple[Any, ...]:
        return getattr(self.base, "shape", ())

    @property
    def dtype(self) -> DType | None:
        return getattr(self.base, "dtype", None)

    def __getitem__(self, key: Any) -> "BufferRegion":
        return BufferRegion(base=self, indices=normalize_indices(key))


@dataclass(frozen=True, slots=True)
class ProcessStyleCall:
    """Captured process-style operation placeholder."""

    op: str
    args: tuple[Any, ...] = ()
    out: Any = None
    attrs: dict[str, Any] = field(default_factory=dict)

    @property
    def shape(self) -> tuple[Any, ...]:
        return getattr(self.out, "shape", ())


@dataclass(frozen=True, slots=True)
class SerialRange:
    """Simulator-visible serial loop descriptor."""

    start: Any
    stop: Any
    step: Any

    def __iter__(self) -> Iterable[int]:
        if any(is_expr(value) for value in (self.start, self.stop, self.step)):
            raise TypeError("A.Serial with symbolic bounds can only be consumed by the atlang AST parser.")
        return iter(builtins.range(int(self.start), int(self.stop), int(self.step)))


@dataclass(frozen=True, slots=True)
class CoreArray:
    """Top-level core-array context placeholder."""

    shape: tuple[Any, ...]
    attrs: dict[str, Any] = field(default_factory=dict)

    def __init__(self, shape: Any, **kwargs: Any) -> None:
        normalized_shape = normalize_shape(shape)
        if not normalized_shape:
            raise ValueError("CoreArray shape must be non-empty.")
        object.__setattr__(self, "shape", normalized_shape)
        object.__setattr__(self, "attrs", dict(kwargs))

    def __enter__(self) -> "CoreArray":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> builtins.bool:
        return False


@dataclass(frozen=True, slots=True)
class SPMD:
    """SPMD operator-region context placeholder."""

    name: str
    type: str
    kwargs: dict[str, Any] = field(default_factory=dict)

    def __init__(self, name: str, type: str, **kwargs: Any) -> None:
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "type", type)
        object.__setattr__(self, "kwargs", dict(kwargs))

    def __enter__(self) -> Any:
        if self.type == "gemm":
            return (symbol(f"{self.name}.core_M"), symbol(f"{self.name}.core_K"), symbol(f"{self.name}.core_N"))
        if self.type == "decode-attention":
            return (
                symbol(f"{self.name}.core_R"),
                symbol(f"{self.name}.core_KV"),
                symbol(f"{self.name}.core_N"),
                symbol(f"{self.name}.core_S"),
                symbol(f"{self.name}.core_H"),
            )
        if self.type == "general":
            return None
        raise ValueError(f"Unsupported SPMD type {self.type!r}.")

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> builtins.bool:
        return False


@dataclass(frozen=True, slots=True)
class MPMD:
    """MPMD operator-region context placeholder."""

    name: str
    type: str
    kwargs: dict[str, Any] = field(default_factory=dict)

    def __init__(self, name: str, type: str, **kwargs: Any) -> None:
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "type", type)
        object.__setattr__(self, "kwargs", dict(kwargs))

    def __enter__(self) -> Any:
        return symbol("CORE_ID")

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> builtins.bool:
        return False


@dataclass(frozen=True, slots=True)
class Kernel:
    """Kernel-launch context placeholder."""

    blocks: tuple[Any, ...]
    autotune: builtins.bool | None = None
    core_list: tuple[Any, ...] | None = None

    def __init__(
        self,
        *blocks: Any,
        autotune: builtins.bool | None = None,
        core_list: Any = None,
        **kwargs: Any,
    ) -> None:
        reject_kwargs("Kernel", kwargs)
        object.__setattr__(self, "blocks", tuple(blocks))
        object.__setattr__(self, "autotune", autotune)
        object.__setattr__(self, "core_list", None if core_list is None else tuple(core_list))

    def __enter__(self) -> Any:
        if len(self.blocks) == 0:
            return None
        block_symbols = tuple(symbol(f"block_{index}") for index in builtins.range(len(self.blocks)))
        if len(block_symbols) == 1:
            return block_symbols[0]
        return block_symbols

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> builtins.bool:
        return False


__all__ = [
    "Buffer",
    "BufferRegion",
    "CoreArray",
    "Kernel",
    "MPMD",
    "ProcessStyleCall",
    "SPMD",
    "SerialRange",
    "Tensor",
    "TensorRegion",
]
