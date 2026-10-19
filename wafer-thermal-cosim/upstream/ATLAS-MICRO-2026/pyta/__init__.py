"""Package entry point for migrated PyTA modules."""

from pathlib import Path

_IMPL_DIR = Path(__file__).resolve().parent / "pyta"

if _IMPL_DIR.is_dir():
    __path__ = [*__path__, str(_IMPL_DIR)]

del Path, _IMPL_DIR
