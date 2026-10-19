"""Standalone atlang frontend package."""

from . import language as language
from .ir import *  # noqa: F403
from .ir import __all__ as _ir_all
from .language import *  # noqa: F403
from .language import __all__ as _language_all

__all__ = list(dict.fromkeys(["language", *_ir_all, *_language_all]))
