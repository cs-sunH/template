"""DType registry for the standalone atlang frontend."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


# DType objects

@dataclass(frozen=True, slots=True)
class DType:
    """Canonical scalar dtype descriptor."""

    name: str
    bits: int

    @property
    def byte_size(self) -> int:
        return max((int(self.bits) + 7) // 8, 1)

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return self.name


# Registry construction

def _make_dtype(name: str, bits: int) -> DType:
    return DType(name=name, bits=int(bits))


_CANONICAL_DTYPES: dict[str, DType] = {
    "bool": _make_dtype("bool", 1),
    "int8": _make_dtype("int8", 8),
    "int16": _make_dtype("int16", 16),
    "int32": _make_dtype("int32", 32),
    "int64": _make_dtype("int64", 64),
    "uint8": _make_dtype("uint8", 8),
    "uint16": _make_dtype("uint16", 16),
    "uint32": _make_dtype("uint32", 32),
    "uint64": _make_dtype("uint64", 64),
    "float16": _make_dtype("float16", 16),
    "bfloat16": _make_dtype("bfloat16", 16),
    "float32": _make_dtype("float32", 32),
    "float64": _make_dtype("float64", 64),
    "float8_e4m3": _make_dtype("float8_e4m3", 8),
    "float8_e5m2": _make_dtype("float8_e5m2", 8),
}

_ALIASES: dict[str, str] = {
    "boolean": "bool",
    "float": "float32",
    "double": "float64",
    "half": "float16",
    "fp16": "float16",
    "bf16": "bfloat16",
    "fp32": "float32",
    "fp64": "float64",
    "float8_e4m3fn": "float8_e4m3",
    "float8_e4m3fnuz": "float8_e4m3",
    "float8_e5m2fn": "float8_e5m2",
    "float8_e5m2fnuz": "float8_e5m2",
    "e4m3": "float8_e4m3",
    "e4m3fn": "float8_e4m3",
    "e4m3fnuz": "float8_e4m3",
    "e5m2": "float8_e5m2",
    "e5m2fn": "float8_e5m2",
    "e5m2fnuz": "float8_e5m2",
}


# Lookup helpers

def dtype_of(value: Any) -> DType:
    if isinstance(value, DType):
        return value
    key = str(value).strip()
    canonical_name = _ALIASES.get(key, key)
    dtype = _CANONICAL_DTYPES.get(canonical_name)
    if dtype is None:
        raise ValueError(f"Unsupported atlang dtype {value!r}.")
    return dtype


def dtype_byte_size(value: Any) -> int:
    return dtype_of(value).byte_size


def dtype_names() -> tuple[str, ...]:
    return tuple(_CANONICAL_DTYPES)


# Public constants

bool = _CANONICAL_DTYPES["bool"]
int8 = _CANONICAL_DTYPES["int8"]
int16 = _CANONICAL_DTYPES["int16"]
int32 = _CANONICAL_DTYPES["int32"]
int64 = _CANONICAL_DTYPES["int64"]
uint8 = _CANONICAL_DTYPES["uint8"]
uint16 = _CANONICAL_DTYPES["uint16"]
uint32 = _CANONICAL_DTYPES["uint32"]
uint64 = _CANONICAL_DTYPES["uint64"]
float16 = _CANONICAL_DTYPES["float16"]
bfloat16 = _CANONICAL_DTYPES["bfloat16"]
float32 = _CANONICAL_DTYPES["float32"]
float64 = _CANONICAL_DTYPES["float64"]
float8_e4m3 = _CANONICAL_DTYPES["float8_e4m3"]
float8_e5m2 = _CANONICAL_DTYPES["float8_e5m2"]


__all__ = [
    "DType",
    "bfloat16",
    "bool",
    "dtype_byte_size",
    "dtype_names",
    "dtype_of",
    "float16",
    "float32",
    "float64",
    "float8_e4m3",
    "float8_e5m2",
    "int16",
    "int32",
    "int64",
    "int8",
    "uint16",
    "uint32",
    "uint64",
    "uint8",
]
