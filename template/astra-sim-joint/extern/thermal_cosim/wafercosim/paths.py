"""Locate packaged data (thermal/energy configs) independently of the caller's cwd.

Configurations install as package data under ``wafercosim/configs`` so the
embedded component resolves them without relying on a sibling directory or an
absolute ``PYTHONPATH``.
"""
from __future__ import annotations

from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
CONFIGS_DIR = PACKAGE_ROOT / "configs"


def configs_dir() -> Path:
    """Directory holding the packaged thermal/energy/fixture configs."""
    return CONFIGS_DIR


def config_path(*parts: str) -> Path:
    """Resolve a packaged config path such as ``("thermal", "x.json")``."""
    path = CONFIGS_DIR.joinpath(*parts)
    if not path.is_file():
        raise FileNotFoundError(f"packaged config does not exist: {path}")
    return path


__all__ = ["PACKAGE_ROOT", "CONFIGS_DIR", "configs_dir", "config_path"]
